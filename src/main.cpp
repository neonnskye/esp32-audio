#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUDP.h>
#include "esp_wifi.h"
#include "driver/adc.h"
#include <driver/i2s.h>

#define EIDSP_QUANTIZE_FILTERBANK 0
#include <Elio_Wake_v3_inferencing.h>

// ---- User configuration ----
#define WIFI_SSID "Amrith’s iPhone"
#define WIFI_PASSWORD "brat summer"
#define PC_IP "172.20.10.5"
#define UDP_PORT 12345
#define CTRL_UDP_PORT 12346
#define AUDIO_RX_PORT 12347    // PC sends TTS audio back to this port
#define PLAYBACK_VOLUME_PCT 90 // volume scale applied to each sample (out of 100)
// ----------------------------

#define SAMPLES_PER_PKT 512

// Double buffer: ISR writes to one half, main loop sends the other
uint16_t buf[2][SAMPLES_PER_PKT];
volatile int writeBuf = 0;
volatile int writeIdx = 0;
volatile int readyBuf = -1;

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
hw_timer_t *timer = NULL;

volatile bool isSpeaking = false;

// EI inference double-buffer (fed from the timer ISR, consumed by inference task)
typedef struct
{
    int16_t *buffers[2];
    volatile uint8_t buf_select;
    volatile uint8_t buf_ready;
    volatile uint32_t buf_count;
    uint32_t n_samples;
} ei_inference_t;

static ei_inference_t ei_inf;
static TaskHandle_t inferenceTaskHandle = NULL;

// ISR: fires at exactly 16 000 Hz, reads one ADC sample, swaps buffer when full.
// adc1_get_raw() is called outside the spinlock to keep the critical section minimal.
void IRAM_ATTR onTimer()
{
    uint16_t sample = (uint16_t)adc1_get_raw(ADC1_CHANNEL_7);

    // Do arithmetic OUTSIDE the lock
    // (cheap optimization, avoids extra work while holding spinlock)
    int16_t s16 = (int16_t)(sample - 2048) * 16;

    portENTER_CRITICAL_ISR(&mux);

    // UDP buffer
    buf[writeBuf][writeIdx++] = sample;
    if (writeIdx >= SAMPLES_PER_PKT)
    {
        writeIdx = 0;
        readyBuf = writeBuf;
        writeBuf ^= 1;
    }

    // EI inference buffer
    ei_inf.buffers[ei_inf.buf_select][ei_inf.buf_count++] = s16;

    if (ei_inf.buf_count >= ei_inf.n_samples)
    {
        ei_inf.buf_select ^= 1;
        ei_inf.buf_count = 0;

        // Notify inference task instead of setting a flag
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        vTaskNotifyGiveFromISR(inferenceTaskHandle, &xHigherPriorityTaskWoken);

        if (xHigherPriorityTaskWoken)
        {
            portYIELD_FROM_ISR();
        }
    }

    portEXIT_CRITICAL_ISR(&mux);
}

static int ei_get_data(size_t offset, size_t length, float *out_ptr)
{
    uint8_t done_buf = ei_inf.buf_select ^ 1;
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (float)ei_inf.buffers[done_buf][offset + i] / 2048.0f;
    }
    return 0;
}

static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

WiFiUDP udp;
WiFiUDP ctrlUdp;

void inferenceTask(void *arg)
{
    static bool debug_nn = false;
    static uint32_t ledOffAt = 0;

    while (true)
    {
        // Block indefinitely until the ISR sends a notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Turn off LED if blink duration has elapsed (non-blocking)
        if (ledOffAt > 0 && millis() >= ledOffAt)
        {
            digitalWrite(LED_BUILTIN, LOW);
            ledOffAt = 0;
        }

        signal_t signal;
        signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
        signal.get_data = &ei_get_data;

        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK)
        {
            Serial.printf("EI classifier error: %d\n", err);
            continue;
        }

        if (++print_results >= EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)
        {
            for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                Serial.printf("  %s: %.2f\n",
                              result.classification[i].label,
                              result.classification[i].value);

                if (strcmp(result.classification[i].label, "elio") == 0 &&
                    result.classification[i].value > 0.6f)
                {
                    Serial.println(">>> WAKE WORD DETECTED <<<");
                    digitalWrite(LED_BUILTIN, HIGH);
                    ledOffAt = millis() + 500; // schedule off, don't block

                    // Notify Python server that wake word was detected
                    // Suppress the trigger while ESP32 is playing its own audio
                    // (prevents acoustic feedback through the microphone)
                    if (!isSpeaking)
                    {
                        uint8_t trigByte = 0x01;
                        ctrlUdp.beginPacket(PC_IP, CTRL_UDP_PORT);
                        ctrlUdp.write(&trigByte, 1);
                        ctrlUdp.endPacket();
                    }
                }
            }
            print_results = 0;
        }
    }
}

void audioPlaybackTask(void *arg)
{
    WiFiUDP audioRxUdp;
    audioRxUdp.begin(AUDIO_RX_PORT);

    int16_t rxBuf[SAMPLES_PER_PKT];
    int16_t stereoBuf[SAMPLES_PER_PKT * 2];
    uint32_t lastPacketMs = 0;

    while (true)
    {
        int packetSize = audioRxUdp.parsePacket();

        if (packetSize > 0)
        {
            int bytesRead = audioRxUdp.read((uint8_t *)rxBuf, sizeof(rxBuf));
            int samplesRead = bytesRead / sizeof(int16_t);

            isSpeaking = true;
            lastPacketMs = millis();

            for (int i = 0; i < samplesRead; i++)
            {
                int16_t s = (int16_t)((int32_t)rxBuf[i] * PLAYBACK_VOLUME_PCT / 100);
                stereoBuf[i * 2] = s;
                stereoBuf[i * 2 + 1] = s;
            }

            size_t bytesWritten;
            i2s_write(I2S_NUM_0, stereoBuf, samplesRead * 4, &bytesWritten, portMAX_DELAY);
        }
        else
        {
            if (isSpeaking && (millis() - lastPacketMs > 200))
            {
                isSpeaking = false;
            }
            delay(1);
        }
    }
}

uint32_t packetsSent = 0;
uint32_t packetsFailed = 0;

void setup()
{
    Serial.begin(115200);

    // Configure ADC via IDF API (GPIO 35 = ADC1 channel 7)
    // 12-bit resolution, 11 dB attenuation = full 0–3.3 V input range
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_7, ADC_ATTEN_DB_12);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected, IP: ");
    Serial.println(WiFi.localIP());

    // Give lwIP time to populate ARP table and prepare UDP send buffers
    Serial.println("Waiting for network to stabilize...");
    delay(2000);

    // Disable WiFi modem sleep to reduce ADC interference from radio bursts
    esp_wifi_set_ps(WIFI_PS_NONE);

    ei_inf.n_samples = EI_CLASSIFIER_SLICE_SIZE;
    ei_inf.buffers[0] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    ei_inf.buffers[1] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    ei_inf.buf_select = 0;
    ei_inf.buf_count = 0;
    ei_inf.buf_ready = 0;

    run_classifier_init();

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Initialize I2S for audio playback (MAX98357 amp)
    // 16-bit 16 kHz mono, duplicated to both stereo channels
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = SAMPLES_PER_PKT,
        .use_apll = false,
        .tx_desc_auto_clear = true,
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = 26,
        .ws_io_num = 25,
        .data_out_num = 22,
        .data_in_num = I2S_PIN_NO_CHANGE,
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_zero_dma_buffer(I2S_NUM_0);

    // Start inference task on core 0 (WiFi/UDP runs on core 1 by default)
    xTaskCreatePinnedToCore(inferenceTask, "EI_Infer", 1024 * 48, NULL, 1, &inferenceTaskHandle, 0);

    // Start audio playback task on core 1 (same as WiFi — spends most time blocked on UDP recv)
    xTaskCreatePinnedToCore(audioPlaybackTask, "AudioRX", 1024 * 8, NULL, 2, NULL, 1);

    // Timer 0: 80 MHz / prescaler 5 = 16 MHz base clock
    // Alarm at 1000 counts → 16 MHz / 1000 = exactly 16 000 Hz
    timer = timerBegin(0, 5, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 1000, true);
    timerAlarmEnable(timer);

    Serial.println("Streaming audio...");
    Serial.print("Sent: ");
    Serial.print(packetsSent);
    Serial.print(" | Failed: ");
    Serial.println(packetsFailed);
}

void loop()
{
    if (readyBuf < 0)
    {
        delay(1);
        return;
    }

    // Atomically claim and clear readyBuf before doing any UDP work
    int toSend;
    portENTER_CRITICAL(&mux);
    toSend = readyBuf;
    readyBuf = -1; // clear immediately so ISR can reuse the slot
    portEXIT_CRITICAL(&mux);

    if (toSend < 0)
        return; // another core beat us here (defensive)

    if (udp.beginPacket(PC_IP, UDP_PORT) == 0)
    {
        packetsFailed++;
        delay(5);
        return;
    }

    udp.write((uint8_t *)buf[toSend], SAMPLES_PER_PKT * sizeof(uint16_t));

    if (udp.endPacket() != 0)
    {
        packetsSent++;
    }
    else
    {
        packetsFailed++;
        delay(5);
    }

    static uint32_t lastPrint = 0;
    if (millis() - lastPrint > 5000)
    {
        Serial.print("Sent: ");
        Serial.print(packetsSent);
        Serial.print(" | Failed: ");
        Serial.println(packetsFailed);
        lastPrint = millis();
    }
}