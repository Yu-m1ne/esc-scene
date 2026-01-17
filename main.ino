// === 必须放在最前面：关闭 WiFi/BT 以释放 DRAM ===
#include <WiFi.h>
#include <esp_bt.h>

// 禁用 Arduino 内部的 WiFi/BT 启动以节省内存
#define ARDUINO_DISABLE_WIFI
#define ARDUINO_DISABLE_BLUETOOTH

#include <ArduTFLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <math.h>

#include "model_int8.h"

// === OLED Config ===
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_MOSI 23
#define OLED_CLK 18
#define OLED_DC 4
#define OLED_CS 2
#define OLED_RESET 5

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &SPI, OLED_DC, OLED_RESET, OLED_CS);

// === Audio & DSP Config ===
const int SAMPLE_RATE = 16000;
const int N_FFT = 512;        
const int HOP_LENGTH = 160;   
const int N_MELS_CALC = 80;   
const float FMIN = 0.0f;      
const float FMAX = 8000.0f;   

const int MODEL_H = 32;  // Time steps
const int MODEL_W = 32;  // Input Features

// 增益控制
#define INPUT_GAIN 2.0f 
#define HISTORY_SIZE 3

// === 内存优化：稀疏滤波器索引 ===
int16_t* mel_start_indices = nullptr;
int16_t* mel_stop_indices = nullptr;

// === Global Buffers ===
float* hamming_window = nullptr;
int16_t* i2s_read_buff = nullptr;  
float* fft_input_buff = nullptr;   
float* fft_output_mag = nullptr;   

const char* LABELS[] = { "Crying", "Sneeze", "Cough", "Noise" };
const int NUM_CLASSES = 4;

// === TFLite Globals ===
namespace {
    tflite::MicroMutableOpResolver<24> op_resolver;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;

    // 【修改点】增加 Arena 大小到 80KB，确保 allocate tensors 成功
    const int kTensorArenaSize = 20 * 1024; 
    uint8_t* tensor_arena = nullptr; 
}

// === Function Declarations ===
bool initSystem();
bool setupI2SMic();
void initHammingWindow();
void initMelFilterIndices(); 
void computeFastFFT(float* v_real, float* v_imag, uint16_t samples);
void processStreamAndInfer();
void softmax(float* values, int size);

// ==================== SETUP ====================
void setup() {
    btStop();
    WiFi.mode(WIFI_OFF);
    esp_bt_controller_disable();

    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== Final Audio Classifier ===");

    if (!initSystem()) {
        Serial.println("FATAL: System Init Failed");
        while (1) delay(100);
    }
}

void loop() {
    processStreamAndInfer();
    delay(2); 
}

// ==================== SYSTEM INIT ====================
bool initSystem() {
    SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
    if (!display.begin(SSD1306_SWITCHCAPVCC)) {
        Serial.println("❌ OLED Failed");
        return false;
    }
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Init...");
    display.display();

    // 1. 分配模型 Arena (尝试使用 SPIRAM，没有则用 Internal)
    Serial.printf("1. Allocating Arena (%d KB)...\n", kTensorArenaSize / 1024);
    
    // 优先尝试外部 PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (tensor_arena == nullptr) {
        Serial.println("  > PSRAM not found, using internal RAM...");
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT);
    }

    if (tensor_arena == nullptr) {
        Serial.println("❌ Arena Failed! Not enough RAM.");
        return false;
    }
    Serial.println("✅ Arena OK!");
    delay(20);

    // 2. 分配 DSP 缓冲区
    Serial.println("2. Allocating DSP Buffers...");
    
    hamming_window = (float*)heap_caps_calloc(N_FFT, sizeof(float), MALLOC_CAP_DEFAULT);
    mel_start_indices = (int16_t*)heap_caps_calloc(N_MELS_CALC, sizeof(int16_t), MALLOC_CAP_DEFAULT);
    mel_stop_indices = (int16_t*)heap_caps_calloc(N_MELS_CALC, sizeof(int16_t), MALLOC_CAP_DEFAULT);
    i2s_read_buff = (int16_t*)heap_caps_calloc(HOP_LENGTH, sizeof(int16_t), MALLOC_CAP_8BIT);
    fft_input_buff = (float*)heap_caps_calloc(N_FFT, sizeof(float), MALLOC_CAP_8BIT);
    fft_output_mag = (float*)heap_caps_calloc(N_FFT / 2 + 1, sizeof(float), MALLOC_CAP_DEFAULT);

    if (!hamming_window || !mel_start_indices || !mel_stop_indices || !i2s_read_buff || !fft_input_buff || !fft_output_mag) {
        Serial.println("❌ DSP Malloc Failed");
        return false;
    }
    Serial.println("✅ DSP OK!");

    initHammingWindow();
    initMelFilterIndices(); 

    if (!setupI2SMic()) return false;

    // 3. 加载模型
    Serial.println("3. Loading Model...");
    model = tflite::GetModel(ward_model_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Schema Error: %d\n", model->version());
        return false;
    }

    op_resolver.AddConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddAveragePool2D(); // 必须有这个
    op_resolver.AddMean();          // 建议加上这个，GAP 有时会转为 Mean
    op_resolver.AddFullyConnected();
    op_resolver.AddRelu();
    op_resolver.AddReshape();
    op_resolver.AddSoftmax();
    op_resolver.AddDequantize();
    op_resolver.AddQuantize();
    op_resolver.AddAdd();
    op_resolver.AddMul();
    
    // 之前添加的修补算子
    op_resolver.AddStridedSlice();
    op_resolver.AddPad();
    op_resolver.AddConcatenation();
    op_resolver.AddPack();
    op_resolver.AddTranspose();
    
    static tflite::MicroInterpreter interp(model, op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &interp;

    // 这一步之前失败了，现在加大了 Arena 应该能通过
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ AllocateTensors Failed! Need bigger Arena?");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // 打印成功信息
    Serial.print("✅ Model Loaded! Input size: ");
    Serial.println(input_tensor->bytes);

    display.clearDisplay();
    display.println("Listening...");
    display.display();
    Serial.println("=== SYSTEM READY ===");
    return true;
}

// ==================== I2S ====================
bool setupI2SMic() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT, 
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 160, 
        .use_apll = false
    };
    i2s_pin_config_t pins = {
        .bck_io_num = 26,
        .ws_io_num = 25,
        .data_out_num = -1,
        .data_in_num = 35
    };
    if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) return false;
    if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
    i2s_start(I2S_NUM_0);
    return true;
}

// ==================== DSP Logic ====================
void initHammingWindow() {
    for (int i = 0; i < N_FFT; i++) hamming_window[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / (N_FFT - 1));
}

void initMelFilterIndices() {
    auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };
    
    float mel_min = hz_to_mel(FMIN);
    float mel_max = hz_to_mel(FMAX);
    
    float mel_points[N_MELS_CALC + 2];
    for (int i = 0; i < N_MELS_CALC + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (N_MELS_CALC + 1);
    }

    for (int m = 0; m < N_MELS_CALC; m++) {
        float hz_start = mel_to_hz(mel_points[m]);
        float hz_stop = mel_to_hz(mel_points[m+2]); 
        
        mel_start_indices[m] = (int)(hz_start * N_FFT / SAMPLE_RATE);
        mel_stop_indices[m] = (int)(hz_stop * N_FFT / SAMPLE_RATE);
        
        if (mel_start_indices[m] < 0) mel_start_indices[m] = 0;
        if (mel_stop_indices[m] > N_FFT/2) mel_stop_indices[m] = N_FFT/2;
    }
}

void computeFastFFT(float* v_real, float* v_imag, uint16_t samples) {
    uint16_t j = 0;
    for (uint16_t i = 0; i < samples - 1; i++) {
        if (i < j) {
            float tr = v_real[i]; v_real[i] = v_real[j]; v_real[j] = tr;
            float ti = v_imag[i]; v_imag[i] = v_imag[j]; v_imag[j] = ti;
        }
        uint16_t k = samples >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
    for (uint16_t m = 1; m < samples; m <<= 1) {
        float wm_r = cosf(PI / m);
        float wm_i = -sinf(PI / m);
        for (uint16_t k = 0; k < samples; k += (m << 1)) {
            float w_r = 1.0f, w_i = 0.0f;
            for (uint16_t n = 0; n < m; n++) {
                uint16_t idx1 = k + n; uint16_t idx2 = k + n + m;
                float t_r = w_r * v_real[idx2] - w_i * v_imag[idx2];
                float t_i = w_r * v_imag[idx2] + w_i * v_real[idx2];
                float u_r = v_real[idx1]; float u_i = v_imag[idx1];
                v_real[idx1] = u_r + t_r; v_imag[idx1] = u_i + t_i;
                v_real[idx2] = u_r - t_r; v_imag[idx2] = u_i - t_i;
                float twr = w_r * wm_r - w_i * wm_i;
                w_i = w_r * wm_i + w_i * wm_r; w_r = twr;
            }
        }
    }
}

// ==================== PROCESS ====================
void processStreamAndInfer() {
    static float rolling_mel_buffer[MODEL_H * MODEL_W]; // 32x32 buffer
    static bool buffer_filled = false;
    static int fill_counter = 0;
    static float history_scores[HISTORY_SIZE][NUM_CLASSES];
    static int history_idx = 0;
    static float dc_offset = 0.0f;
    
    // 16次累积 ≈ 160ms (匹配 Python 处理的时间密度)
    const int ACCUMULATION_STEPS = 16; 
    
    float mel_energies_accum[N_MELS_CALC] = {0}; 
    int32_t batch_mic_max = 0;

    // =========================================================================
    // 1. 信号采集与功率谱计算
    // =========================================================================
    for (int step = 0; step < ACCUMULATION_STEPS; step++) {
        size_t bytes_read = 0;
        int32_t raw_samples[HOP_LENGTH];
        
        // 读取 I2S 数据
        esp_err_t result = i2s_read(I2S_NUM_0, raw_samples, sizeof(raw_samples), &bytes_read, 100 / portTICK_PERIOD_MS);
        if (result != ESP_OK || bytes_read == 0) return;

        // 移动 FFT 输入缓冲
        memmove(fft_input_buff, &fft_input_buff[HOP_LENGTH], (N_FFT - HOP_LENGTH) * sizeof(float));
        
        // 【修改点 1】降低增益，匹配训练时的归一化电平 (从 60.0 改为 4.0)
        float effective_gain = 4.0f; 

        for (int i = 0; i < HOP_LENGTH; i++) {
            int32_t sample = raw_samples[i] >> 8; // 32bit -> 24bit 有效位
            if (abs(sample) > batch_mic_max) batch_mic_max = abs(sample);
            
            // 简单的 DC 偏移滤除
            dc_offset = 0.95f * dc_offset + 0.05f * (float)sample;
            
            // 归一化到 -1.0 ~ 1.0
            float f_val = ((float)sample - dc_offset) / 8388608.0f * effective_gain;
            fft_input_buff[N_FFT - HOP_LENGTH + i] = fmaxf(-1.0f, fminf(1.0f, f_val));
        }

        // FFT 计算
        static float v_real[N_FFT];
        static float v_imag[N_FFT];
        for (int i=0; i<N_FFT; i++) {
            v_real[i] = fft_input_buff[i] * hamming_window[i];
            v_imag[i] = 0.0f;
        }
        computeFastFFT(v_real, v_imag, N_FFT);
        
        // 计算功率谱 Power Spectrum (|Mag|^2)
        for (int i=0; i <= N_FFT/2; i++) {
            fft_output_mag[i] = v_real[i]*v_real[i] + v_imag[i]*v_imag[i];
        }

        // Mel 滤波器组积分
        auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };
        float mel_min = hz_to_mel(FMIN);
        float mel_max = hz_to_mel(FMAX);
        float mel_step = (mel_max - mel_min) / (N_MELS_CALC + 1);

        for (int m=0; m < N_MELS_CALC; m++) {
            float energy = 0.0f;
            int k_start = mel_start_indices[m];
            int k_stop = mel_stop_indices[m];
            
            float mel_center = mel_min + (m + 1) * mel_step;
            float hz_center = mel_to_hz(mel_center);
            float hz_left = mel_to_hz(mel_min + m * mel_step);
            float hz_right = mel_to_hz(mel_min + (m + 2) * mel_step);

            for (int k = k_start; k < k_stop; k++) {
                float freq = k * (float)SAMPLE_RATE / N_FFT;
                float weight = (freq <= hz_center) ? 
                               (freq - hz_left) / (hz_center - hz_left) : 
                               (hz_right - freq) / (hz_right - hz_center);
                energy += fft_output_mag[k] * weight;
            }
            mel_energies_accum[m] += energy;
        }
    } 

    // =========================================================================
    // 2. Log 处理 (dB) 与 缓冲移位
    // =========================================================================
    float mel_energies_final[N_MELS_CALC];
    float avg_db_debug = 0.0f; 

    for (int m=0; m < N_MELS_CALC; m++) {
        float avg_energy = mel_energies_accum[m] / ACCUMULATION_STEPS;
        
        // Power to dB: 10 * log10
        float db = 10.0f * log10f(fmaxf(avg_energy, 1e-10f));
        
        // 简单的底部截断，防止负无穷
        mel_energies_final[m] = fmaxf(-80.0f, db); 
        
        avg_db_debug += mel_energies_final[m];
    }
    avg_db_debug /= N_MELS_CALC;

    // 滚动缓冲区：将旧数据左移 (丢弃最旧的一列时间步)
    // Buffer 布局假设为 [Time][Freq] 也就是 [MODEL_H][MODEL_W]
    // 但为了匹配之前的逻辑，这里是把 MODEL_W 当作 Mel 频带数量(32)，MODEL_H 当作时间步(32)
    memmove(rolling_mel_buffer, &rolling_mel_buffer[MODEL_W], (MODEL_H - 1) * MODEL_W * sizeof(float));
    
    // 指向最新的一行 (最后一行)
    float* last_row = &rolling_mel_buffer[(MODEL_H - 1) * MODEL_W];
    
    // Resize Logic (从 80 个 Mel 压缩到 32 个输入特征)
    // 简单的平均池化
    for (int w = 0; w < MODEL_W; w++) {
        int start = (w * N_MELS_CALC) / MODEL_W;
        int end = ((w + 1) * N_MELS_CALC) / MODEL_W;
        if (end <= start) end = start + 1;
        float sum = 0;
        for (int k=start; k<end; k++) sum += mel_energies_final[k];
        last_row[w] = sum / (end - start);
    }

    // 缓冲区预热
    if (!buffer_filled) {
        fill_counter++;
        if (fill_counter % 2 == 0) {
            display.clearDisplay();
            display.setCursor(0, 20);
            display.printf("Buffering %d/%d", fill_counter, MODEL_H);
            display.display();
        }
        if (fill_counter >= MODEL_H) buffer_filled = true;
        return;
    }

    // =========================================================================
    // 3. 量化与填充 (【关键修复】使用模型参数而非动态拉伸)
    // =========================================================================
    
    float input_scale = input_tensor->params.scale;
    int32_t input_zero_point = input_tensor->params.zero_point;

    // 【修改点 2】在循环外定义变量，解决编译报错
    int min_q = 127;
    int max_q = -128;

    for (int freq_y = 0; freq_y < MODEL_W; freq_y++) {
        for (int time_x = 0; time_x < MODEL_H; time_x++) {
            
            int buffer_idx = time_x * MODEL_W + freq_y; // Source: [Time][Freq]
            float real_db_val = rolling_mel_buffer[buffer_idx];

            // Target: TFLite Input [Freq][Time] (转置，如果模型需要)
            // 注意：具体需不需要转置取决于 Python 导出时的 permute。
            // 你的 Python 代码里做了 np.transpose(input_data, (0, 2, 3, 1))，通常意味着 NHWC。
            // 这里维持你原代码的映射逻辑：freq_y * MODEL_H + time_x
            int tensor_idx = freq_y * MODEL_H + time_x; 
            
            // 【修改点 3】标准量化公式
            // int8 = (real_value / scale) + zero_point
            int32_t q_val = (int32_t)(real_db_val / input_scale) + input_zero_point;
            
            // Clamp to int8
            int8_t final_val = (int8_t)max(-128, min(127, q_val));
            
            input_tensor->data.int8[tensor_idx] = final_val;

            // 统计范围供调试
            if (final_val < min_q) min_q = final_val;
            if (final_val > max_q) max_q = final_val;
        }
    }

    // 推理
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke Failed");
        return;
    }

    // =========================================================================
    // 4. 结果处理与显示
    // =========================================================================
    float current_scores[NUM_CLASSES];
    float out_scale = output_tensor->params.scale;
    int out_zero_point = output_tensor->params.zero_point;
    
    // 反量化输出
    for (int i = 0; i < NUM_CLASSES; i++) {
        current_scores[i] = (output_tensor->data.int8[i] - out_zero_point) * out_scale;
    }
    softmax(current_scores, NUM_CLASSES);

    // 平滑历史记录
    for(int i=0; i<NUM_CLASSES; i++) history_scores[history_idx][i] = current_scores[i];
    history_idx = (history_idx + 1) % HISTORY_SIZE;

    float avg_scores[NUM_CLASSES] = {0};
    int top_idx = 0;
    float max_avg = 0;
    
    for(int i=0; i<NUM_CLASSES; i++) {
        for(int h=0; h<HISTORY_SIZE; h++) avg_scores[i] += history_scores[h][i];
        avg_scores[i] /= HISTORY_SIZE;
        if (avg_scores[i] > max_avg) {
            max_avg = avg_scores[i];
            top_idx = i;
        }
    }

    // 【修改点 4】置信度与底噪过滤
    // 如果声音太小 (-60dB) 或者 模型信心不足 (40%)，强制判定为 Noise
    // 注意：这里的 -60dB 取决于上面的 dB 计算，如果你发现安静时是 -50，这里就改 -55
    bool is_uncertain = false;
    if (max_avg < 0.40f || avg_db_debug < -60.0f) {
        top_idx = 3; // 假设 Index 3 是 Noise/Background
        is_uncertain = true;
    }

    // 串口打印
    Serial.printf("Best: %s (%.0f%%) | Mic: %d | dB: %.1f | Range: [%d, %d]\n", 
        LABELS[top_idx], max_avg*100, batch_mic_max, avg_db_debug, min_q, max_q);

    // OLED 显示
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(2);
    
    if (is_uncertain && max_avg < 0.40f) {
        // 如果是因为信心不足，可以显示个问号或者 Unsure
        display.println("Unsure"); 
    } else {
        // 正常显示类别
        if (top_idx == 3) display.println("Noise"); // 可以把 Background 显示为 Noise
        else display.println(LABELS[top_idx]);
    }

    display.setTextSize(1);
    display.setCursor(0, 35);
    display.printf("Conf: %d%%", (int)(max_avg * 100));
    
    // 画信心条
    display.drawRect(60, 35, 64, 8, SSD1306_WHITE);
    display.fillRect(62, 37, (int)(60 * max_avg), 4, SSD1306_WHITE);

    // 画音量条 (简单的可视化)
    int vol_bar_width = map(min(batch_mic_max, (int32_t)10000), 0, 10000, 0, 128);
    display.fillRect(0, 56, vol_bar_width, 4, SSD1306_WHITE);
    
    display.display();
}

void softmax(float* v, int n) {
    float max_val = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_val) max_val = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }
    for (int i = 0; i < n; i++) v[i] /= sum;
}