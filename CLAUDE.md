# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Flash Commands

All commands run via PlatformIO CLI or the PlatformIO IDE extension.

```bash
# Build
pio run

# Build and upload to device
pio run --target upload

# Monitor serial output (115200 baud)
pio device monitor

# Build, upload, and monitor in one step
pio run --target upload && pio device monitor

# Clean build artifacts
pio run --target clean
```

## Project Overview

ESP32 audio project targeting the **NodeMCU-32S** board using the Arduino framework via PlatformIO.
The firmware streams real-time audio from a MAX9814 analog microphone over Wi-Fi UDP to a Python backend for playback and processing.

- **Target board:** `nodemcu-32s` (Espressif ESP32)
- **Framework:** Arduino
- **Serial baud rate:** 115200
- **No external library dependencies** — uses only the built-in Arduino ESP32 framework (`WiFi.h`, `WiFiUDP.h`) and ESP-IDF headers (`driver/adc.h`, `esp_wifi.h`)

## Hardware

- **Microphone:** MAX9814 analog electret mic amplifier
- **ADC pin:** GPIO 35 (ADC1 channel 7, input-only)
- **ADC config:** 12-bit resolution, 11 dB attenuation (full 0–3.3 V range)
- **Sample rate:** 16 000 Hz

## Code Structure

- [src/main.cpp](src/main.cpp) — single entry point; Arduino `setup()` and `loop()`
- [platformio.ini](platformio.ini) — board, platform, and framework config
- [receiver.py](receiver.py) — Python UDP receiver and real-time audio playback
- [pyproject.toml](pyproject.toml) — Python project config (managed with `uv`)
- [lib/](lib/) — local libraries (currently empty)
- [include/](include/) — shared headers (currently empty)
- `.pio/` — generated build artifacts and downloaded lib dependencies (not edited manually)
- `.venv/` — Python virtual environment (managed by `uv`, not edited manually)

## Current Functionality

The firmware samples the MAX9814 microphone at **16 kHz** using a hardware timer ISR and streams raw ADC samples to a PC over **Wi-Fi UDP**.

### Architecture

- **Hardware timer ISR (`onTimer`)** — fires at exactly 16 000 Hz (timer 0, prescaler 5, alarm 1000; derived from 80 MHz CPU clock). Each invocation calls `adc1_get_raw(ADC1_CHANNEL_7)` and stores the 12-bit sample into the active half of a double buffer. When 512 samples are collected the buffer is marked ready and the write pointer swaps to the other half.
- **`loop()`** — when a buffer is flagged ready, sends the 1024-byte payload (512 × `uint16_t`) as a single UDP packet to the configured PC IP, then calls `yield()` to allow the lwIP stack to drain its send queue.
- **Double buffer** — decouples sampling from sending so the ISR never stalls waiting for UDP transmission.
- **`esp_wifi_set_ps(WIFI_PS_NONE)`** — disables WiFi modem sleep to reduce RF interference on the ADC.

### UDP Packet Format

Each packet is exactly **1024 bytes**: 512 little-endian `uint16_t` samples representing raw 12-bit ADC values (0–4095). The receiver subtracts 2048 (DC midpoint) and normalises to float before playback.

### Configuration (`src/main.cpp` defines)

| Define | Default | Description |
|--------|---------|-------------|
| `WIFI_SSID` | — | Wi-Fi network name |
| `WIFI_PASSWORD` | — | Wi-Fi password |
| `PC_IP` | — | Receiver's IPv4 address (run `ipconfig` on Windows) |
| `UDP_PORT` | `12345` | UDP port (must match Python receiver) |
| `SAMPLES_PER_PKT` | `512` | Samples per UDP packet |

### Python Backend

The receiver lives at [receiver.py](receiver.py) in this repository. It is a two-threaded design:

- **Receive thread (`receive_loop`)** — binds UDP socket on port 12345, receives 1024-byte datagrams, decodes `uint16_t` samples to `float32` (subtract DC offset 2048, divide by 2048), applies a noise gate (mutes packets whose RMS is below `NOISE_GATE = 0.02`), and appends decoded audio to a `deque`. Drops the oldest packet when the queue exceeds `MAX_QUEUE_LEN = 10` (~320 ms of audio).
- **sounddevice callback (`audio_callback`)** — called by `sounddevice` to fill output buffers. Drains the deque into the output array; carries leftover samples across callbacks to avoid boundary gaps. Outputs silence on underrun.
- **Pre-buffering** — playback does not start until `PREBUFFER_PKTS = 3` packets (~96 ms) are queued, reducing the chance of an immediate underrun at startup.

#### Python Configuration (`receiver.py` constants)

| Constant | Default | Description |
|----------|---------|-------------|
| `UDP_PORT` | `12345` | Must match firmware `UDP_PORT` |
| `SAMPLE_RATE` | `16000` | Must match firmware sample rate |
| `SAMPLES_PER_PKT` | `512` | Must match firmware `SAMPLES_PER_PKT` |
| `PREBUFFER_PKTS` | `3` | Packets to queue before playback starts (~96 ms) |
| `MAX_QUEUE_LEN` | `10` | Max queued packets before dropping oldest (~320 ms) |
| `NOISE_GATE` | `0.02` | RMS threshold below which a packet is silenced (0 = off) |

#### Running the receiver

```bash
# Install dependencies (requires uv)
uv sync

# Run
uv run receiver.py
```
