import socket
import collections
import threading
import queue
import time
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ---- Configuration ----
UDP_IP          = "0.0.0.0"  # Listen on all interfaces
UDP_PORT        = 12345
SAMPLE_RATE     = 16000
SAMPLES_PER_PKT = 512
PREBUFFER_PKTS  = 3    # Packets to queue before starting playback (~96ms)
MAX_QUEUE_LEN   = 10   # Drop oldest if queue grows beyond this (~320ms)
NOISE_GATE      = 0.02 # RMS threshold below which a packet is muted (0 = off)
# -----------------------

# Whisper config
WHISPER_MODEL   = "base"       # tiny / base / small — tradeoff speed vs accuracy
WHISPER_DEVICE  = "cpu"        # or "cuda" if you have a GPU
WHISPER_COMPUTE = "int8"       # int8 = fastest on CPU

# VAD / segmentation config
VAD_SILENCE_MS  = 700          # ms of silence before we consider speech done
VAD_MIN_SPEECH_MS = 400        # ignore speech segments shorter than this
MAX_SEGMENT_S   = 10           # hard cap — transcribe even if no silence detected
# -----------------------

packet_queue: collections.deque = collections.deque()
vad_queue: collections.deque = collections.deque()
queue_lock = threading.Lock()
leftover: np.ndarray = np.zeros(0, dtype=np.float32)

# A separate queue to pass completed audio segments to the transcription thread
transcribe_queue: queue.Queue = queue.Queue()

# --- VAD accumulator state ---
accumulator: list[np.ndarray] = []
silence_packets = 0
SILENCE_THRESHOLD   = NOISE_GATE          # reuse your existing threshold
SILENCE_PACKETS_MAX = int((VAD_SILENCE_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MIN_SPEECH_PACKETS  = int((VAD_MIN_SPEECH_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MAX_SEGMENT_PACKETS = int(MAX_SEGMENT_S * SAMPLE_RATE / SAMPLES_PER_PKT)


def vad_accumulator_loop() -> None:
    """
    Watches packet_queue, accumulates audio into speech segments,
    and pushes complete segments onto transcribe_queue.
    """
    global accumulator, silence_packets

    while True:
        # Poll the packet queue — NEVER busy-wait while holding the lock
        chunk = None
        with queue_lock:
            if vad_queue:
                chunk = vad_queue.popleft()

        if chunk is None:
            time.sleep(0.005)  # 5 ms — avoids 100% CPU spin
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = rms >= SILENCE_THRESHOLD

        if rms > 0.001:
            sys.stdout.write(f"\rVAD RMS={rms:.4f} speech={is_speech} acc_len={len(accumulator)} ")
            sys.stdout.flush()

        if is_speech:
            accumulator.append(chunk)
            silence_packets = 0
        else:
            if accumulator:               # we were in speech, now trailing silence
                silence_packets += 1
                accumulator.append(chunk) # include the silence tail (helps Whisper)

        # Dispatch check runs regardless of whether current chunk is speech
        if accumulator:
            end_of_speech = (not is_speech) and (silence_packets >= SILENCE_PACKETS_MAX)
            too_long      = len(accumulator) >= MAX_SEGMENT_PACKETS

            if end_of_speech or too_long:
                segment = np.concatenate(accumulator)
                if len(accumulator) >= MIN_SPEECH_PACKETS:
                    print(f"\n[VAD] Segment ready: {len(accumulator)} packets, {len(segment)} samples, rms={rms:.4f}")
                    transcribe_queue.put(segment)
                else:
                    print(f"\n[VAD] Segment too short ({len(accumulator)} < {MIN_SPEECH_PACKETS}), discarding")
                accumulator = []
                silence_packets = 0

def transcription_loop(model: WhisperModel) -> None:
    """Pulls audio segments from transcribe_queue and runs Whisper."""
    while True:
        segment: np.ndarray = transcribe_queue.get()  # blocks until a segment arrives

        try:
            print(f"[transcribe] Got segment of {len(segment)} samples, transcribing...", flush=True)
            # faster-whisper expects float32 audio at 16kHz — which is exactly what we have
            segments, info = model.transcribe(
                segment,
                language="en",           # set to None for auto-detect
                beam_size=1,             # faster, slightly less accurate
            )

            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                print(f"[transcript] {text}")
        except Exception as exc:
            print(f"[transcribe error] {exc}", flush=True)

def receive_loop(sock: socket.socket) -> None:
    """Background thread: receive UDP packets and enqueue decoded audio."""
    expected_bytes = SAMPLES_PER_PKT * 2  # uint16 = 2 bytes each
    while True:
        data, _ = sock.recvfrom(expected_bytes * 2)
        if len(data) != expected_bytes:
            continue  # drop malformed packets
        raw = np.frombuffer(data, dtype="<u2").astype(np.float32)
        # Remove DC offset (ADC midpoint = 2048) and normalize to [-1.0, 1.0]
        audio = (raw - 2048.0) / 2048.0
        # Noise gate: mute packets below RMS threshold (kills ADC idle noise)
        # Apply to playback queue only — VAD needs the original audio for detection
        if NOISE_GATE > 0 and np.sqrt(np.mean(audio ** 2)) < NOISE_GATE:
            playback_audio = np.zeros(SAMPLES_PER_PKT, dtype=np.float32)
        else:
            playback_audio = audio
        with queue_lock:
            if len(packet_queue) >= MAX_QUEUE_LEN:
                packet_queue.popleft()
            packet_queue.append(playback_audio)
            vad_queue.append(audio)  # <-- original audio (not zeroed)
            if len(vad_queue) > MAX_QUEUE_LEN * 4:
                vad_queue.popleft()


def audio_callback(outdata: np.ndarray, frames: int, time, status) -> None:
    """sounddevice callback: fill outdata with queued audio, silence on underrun."""
    global leftover

    output = np.zeros(frames, dtype=np.float32)
    write_pos = 0
    needed = frames

    # Use leftover samples from previous callback first
    if len(leftover) > 0:
        use = min(len(leftover), needed)
        output[write_pos:write_pos + use] = leftover[:use]
        leftover = leftover[use:]
        write_pos += use
        needed -= use

    # Pull packets from queue until we have enough samples
    while needed > 0:
        with queue_lock:
            if not packet_queue:
                break  # note: should be break, not continue
            chunk = packet_queue.popleft()
        if len(chunk) <= needed:
            output[write_pos:write_pos + len(chunk)] = chunk
            write_pos += len(chunk)
            needed -= len(chunk)
        else:
            # Chunk is larger than remaining space — save the tail for next callback
            output[write_pos:write_pos + needed] = chunk[:needed]
            leftover = chunk[needed:]
            needed = 0

    outdata[:, 0] = output


def main() -> None:
    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("Model loaded.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP audio on port {UDP_PORT}...")

    # Start all background threads
    for target, args in [
        (receive_loop, (sock,)),
        (vad_accumulator_loop, ()),
        (transcription_loop, (model,)),
    ]:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    print(f"Waiting for {PREBUFFER_PKTS} packets to pre-buffer...")
    while True:
        with queue_lock:
            if len(packet_queue) >= PREBUFFER_PKTS:
                break

    print("Starting playback. Press Ctrl+C to stop.")
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=SAMPLES_PER_PKT,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
