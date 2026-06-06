# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Flash Commands

The project targets the NodeMCU-32S board using the Arduino framework. Two options:

### Arduino IDE
- Open `elio-audio.ino` in Arduino IDE
- Install ESP32 board package via Boards Manager (`esp32` by Espressif)
- Select board: **NodeMCU-32S** (or generic `ESP32 DEV Module`)
- Select port and click Upload
- Serial Monitor: 115200 baud

### Arduino CLI (optional)
```bash
# Build
arduino-cli compile --fqbn esp32:esp32:nodemcu-32s elio-audio.ino

# Upload
arduino-cli upload --fqbn esp32:esp32:nodemcu-32s --port <PORT> elio-audio.ino

# Monitor serial output
arduino-cli monitor --port <PORT> --config 115200
```

## Project Overview

ESP32 audio project targeting the **NodeMCU-32S** board using the Arduino framework.
The firmware streams real-time audio from a MAX9814 analog microphone over Wi-Fi UDP.

- **Target board:** `nodemcu-32s` (Espressif ESP32)
- **Framework:** Arduino
- **Serial baud rate:** 115200
- **Dependencies:** built-in Arduino ESP32 framework (`WiFi.h`, `WiFiUDP.h`, `WiFiMulti.h`, `ESPmDNS.h`), ESP-IDF headers (`driver/adc.h`, `driver/i2s.h`, `esp_wifi.h`), `PubSubClient` (MQTT, installed via Arduino Library Manager), and `Elio_Wake_v3_inferencing` (Edge Impulse wake-word model, installed via Arduino Library Manager)

## Hardware

- **Microphone:** MAX9814 analog electret mic amplifier
- **ADC pin:** GPIO 35 (ADC1 channel 7, input-only)
- **ADC config:** 12-bit resolution, 12 dB attenuation (`ADC_ATTEN_DB_12`, full 0–3.3 V range)
- **Sample rate:** 16 000 Hz

## Code Structure

- [elio-audio.ino](elio-audio.ino) — single entry point; Arduino `setup()` and `loop()`
- [jbl_begin.h](jbl_begin.h) — begin chime WAV data (wake word confirmation sound)
- [jbl_latency.h](jbl_latency.h) — latency chime WAV data (STT/LLM gap fill)

## Current Functionality

The firmware samples the MAX9814 microphone at **16 kHz** using a hardware timer ISR and streams raw ADC samples to a PC over **Wi-Fi UDP**.

### Architecture

- **Hardware timer ISR (`onTimer`)** — fires at exactly 16 000 Hz using the v3 timer API (`timerBegin(16000)` — frequency in Hz directly). Each invocation calls `adc1_get_raw(ADC1_CHANNEL_7)` and stores the 12-bit sample into the active half of the UDP double buffer. When 512 samples are collected the buffer is marked ready and the write pointer swaps to the other half.
- **Edge Impulse inference buffer** — same ISR also feeds samples (converted to `int16_t` and upscaled by 16) into a second double buffer managed by `ei_inference_t`. When a slice is full, the ISR sends a FreeRTOS task notification (`vTaskNotifyGiveFromISR`) to wake the inference task.
- **`inferenceTask`** (pinned to core 0) — blocks on `ulTaskNotifyTake` until the ISR signals a slice is ready, then calls `run_classifier_continuous()` and prints per-label scores. When the `"elio"` label exceeds `0.6`:
  1. Sets `isListening = true` (gates subsequent wake word handling)
  2. Turns on `LED_BUILTIN` (stays on until PC sends `"stop"` via MQTT topic `elio/ctrl`)
  3. Publishes `"1"` to MQTT topic `elio/wake` to notify the PC
  4. Plays the begin chime (`jbl_begin.h`) over I2S
  Wake word detection is suppressed while `isSpeaking == true` (self-speech acoustic feedback guard).
- **`loop()`** (core 1) — when a UDP buffer is flagged ready, sends the 1024-byte payload as a single UDP packet to the configured PC IP. Retries on send failure (does not drop packets). Also calls `mqttClient.loop()` for MQTT keep-alive and runs `mqttReconnect()` if disconnected. If `pcIP` is still `0.0.0.0` (Pi not running at boot), calls `resolvePcIP()` every 3 s until the IP is found. Core 1 also hosts the WiFi/UDP/MQTT stack; splitting inference to core 0 prevents EI processing from delaying packet transmission.
- **Double buffer** — decouples sampling from sending so the ISR never stalls waiting for UDP transmission.
- **`WiFiMulti`** — two SSID/password profiles (`WIFI_SSID_1`/`WIFI_PASS_1` and `WIFI_SSID_2`/`WIFI_PASS_2`) provide automatic AP fallback. `setup()` blocks until any configured AP is connected; `loop()` does not monitor WiFi state after that.
- **mDNS host resolution** — the PC (a Raspberry Pi) is discovered via mDNS as `raspberrypi.local` instead of a static IP. `resolvePcIP()` is called at boot (blocking retry loop, 15 attempts × 1 s each) and from `loop()` every 3 s until resolved. Once resolved, `pcIP` is also used as the MQTT broker address; `mqttClient.setServer()` is called with the resolved IP.
- **mDNS self-advertisement** — the ESP32 advertises itself as `esp32-audio.local` so the PC can discover it via mDNS.
- **MQTT reconnection (`mqttReconnect`)** — called from `loop()` whenever the broker is disconnected. Non-blocking: a single `mqttClient.connect()` attempt per call with a 500 ms delay on failure. This keeps `loop()` responsive (no blocking retry loop).
- **`esp_wifi_set_ps(WIFI_PS_NONE)`** — disables WiFi modem sleep to reduce RF interference on the ADC.
- **I2S audio output** — `setup()` initializes I2S in master TX mode (16-bit, 16 kHz stereo) with DMA buffers. Pins: `BCK=26`, `WS=25`, `DATA=22` — drives a MAX98357 I2S amplifier. Mono samples are duplicated to both stereo channels.
- **`audioPlaybackTask`** (core 1) — listens on `AUDIO_RX_PORT` (12347) for TTS audio UDP packets from the PC. Maintains a 3-packet jitter buffer (~96 ms) before starting playback. If a TTS packet arrives while `chimeLooping` is true, it stops the chime loop (sets `chimeLooping = false`, waits for `chimeTaskHandle == NULL`) and zeros the I2S DMA buffer before starting TTS playback. Each received packet (int16 mono) is scaled by `PLAYBACK_VOLUME_PCT`, duplicated to stereo, and written to I2S. A 200 ms silence timeout resets the playback state and signals `isSpeaking = false`.
- **`mqttCallback`** — MQTT subscription callback on topic `elio/ctrl`. Receives string messages from the PC:
  - `"processing"` — starts the looping latency chime (`chimeLoopTask`, plays `jbl_latency.h` in a loop on core 1) to fill the STT/LLM gap with pleasant audio. Guarded by `!isSpeaking` to avoid starting during TTS playback.
  - `"stop"` — stops the latency chime, zeros the I2S DMA buffer, turns off `LED_BUILTIN`, and clears `isListening`.
- **`playChime()`** — plays a WAV chunk (`jbl_begin.h` or `jbl_latency.h`) over I2S. Used for wake-word confirmation (begin chime) and STT/LLM gap fill (latency chime loop). Volume controlled by `CHIME_VOLUME_PCT`.
- **`isSpeaking` gate** — the `inferenceTask` suppresses wake-word detection and MQTT publish when `isSpeaking == true` to prevent acoustic feedback (the ESP32's own speaker audio re-entering the microphone).

### UDP Packet Format

Each packet is exactly **1024 bytes**: 512 little-endian `uint16_t` samples representing raw 12-bit ADC values (0–4095). The receiver subtracts 2048 (DC midpoint) and normalises to float before playback.

### Wake Word Control Channel (MQTT)

When the Edge Impulse classifier detects the wake word (label `"elio"` > 0.6), the ESP32 publishes `"1"` to MQTT topic `TOPIC_WAKE` (`elio/wake`), which drives a `ListenState` state machine on the PC:

| State | Description |
|-------|-------------|
| `IDLE` | Waiting for wake word signal. Audio is streamed but not transcribed. |
| `SKIP_WAKEWORD_BLEED` | Discards `BLEED_SKIP_PACKETS` (~512 ms) of audio after the wake word — covers utterance bleed + begin chime duration. |
| `CAPTURING` | Actively recording the user's command. VAD accumulator builds a speech segment. |
| `TRANSCRIBING` | Groq STT (whisper-large-v3) is transcribing; new captures are blocked. |
| `RESPONDING` | LLM is generating a response / TTS is synthesizing speech; pipeline busy. |

Flow: `IDLE` → (MQTT `elio/wake`: `"1"` received) → `SKIP_WAKEWORD_BLEED` → `CAPTURING` → (silence or max segment) → `TRANSCRIBING` → `RESPONDING` → (TTS playback ends) → `IDLE`.

When entering `CAPTURING`, the PC publishes `"processing"` to MQTT topic `elio/ctrl` — ESP32 starts the looping latency chime (`jbl_latency.h`) to fill the STT/LLM gap with pleasant audio. When TTS audio is ready (or on pipeline abort), the PC publishes `"stop"` to stop the chime, zero the I2S DMA buffer, and turn off the LED.

A `CAPTURE_TIMEOUT_S` (3 s) timer starts when entering `CAPTURING`; if no speech is detected by **Silero VAD** within that window, the state resets to `IDLE` (false-positive guard).

### Configuration (`elio-audio.ino` defines)

| Define | Default | Description |
|--------|---------|-------------|
| `WIFI_SSID_1` / `WIFI_PASS_1` | — | Primary Wi-Fi network credentials |
| `WIFI_SSID_2` / `WIFI_PASS_2` | — | Secondary Wi-Fi network credentials (fallback via `WiFiMulti`) |
| `PC_MDNS_HOST` | `"raspberrypi"` | mDNS hostname of the PC/Raspberry Pi (resolved to IP at runtime) |
| `ESP32_MDNS_HOST` | `"esp32-audio"` | mDNS hostname the ESP32 advertises as |
| `MQTT_BROKER` | `"127.0.0.1"` | MQTT broker address (uses `pcIP` after mDNS resolution) |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_ID` | `"elio-esp32"` | MQTT client ID |
| `TOPIC_WAKE` | `"elio/wake"` | MQTT topic for wake word trigger (payload: `"1"`) |
| `TOPIC_CTRL` | `"elio/ctrl"` | MQTT topic for PC → ESP32 control (`"processing"` / `"stop"`) |
| `UDP_PORT` | `12345` | UDP port for audio stream |
| `AUDIO_RX_PORT` | `12347` | UDP port for receiving TTS audio from PC |
| `PLAYBACK_VOLUME_PCT` | `95` | Volume scale applied to incoming TTS samples (out of 100) |
| `CHIME_VOLUME_PCT` | `95` | Volume scale applied to chime samples (out of 100) |
| `SAMPLES_PER_PKT` | `512` | Samples per UDP packet |

### Python Backend (removed)

The Python receiver (`receiver.py`, tools, and `pyproject.toml`) was deleted in commit `a4531c5`. The firmware remains usable as a standalone UDP audio streamer — any UDP receiver on the PC side can decode the packet format described above.

**Wake word trigger:** ESP32 publishes `"1"` to MQTT topic `TOPIC_WAKE` (`elio/wake`). The PC controls the ESP32's chime/LED via MQTT topic `TOPIC_CTRL` (`elio/ctrl`):
- `"processing"` — start looping latency chime (`jbl_latency.h`)
- `"stop"` — stop chime, zero I2S DMA buffer, turn off LED, reset audio state

### Tools (removed)

All utility scripts in `tools/` were deleted with the Python backend in commit `a4531c5`.

## Troubleshooting

### `endPacket(): could not send data: 12`
Error 12 is `ENOMEM` in lwIP — the UDP send buffer was temporarily unavailable. Root cause was a race where `readyBuf` was marked consumed *before* `endPacket()` was called, so a failed send silently dropped audio data. The fix in `loop()` retries until send succeeds.

### Serial Output During Streaming
ESP32 prints live counts: `Sent: N | Failed: N` every 5 seconds to Serial at **115200 baud**. If `Failed` keeps growing, check network path to receiver. The ESP32 may briefly reset when the serial port connects — this is normal.
