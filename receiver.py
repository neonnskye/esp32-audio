import socket
import collections
import threading

import numpy as np
import sounddevice as sd

# ---- Configuration ----
UDP_IP          = "0.0.0.0"  # Listen on all interfaces
UDP_PORT        = 12345
SAMPLE_RATE     = 16000
SAMPLES_PER_PKT = 512
PREBUFFER_PKTS  = 3    # Packets to queue before starting playback (~96ms)
MAX_QUEUE_LEN   = 10   # Drop oldest if queue grows beyond this (~320ms)
NOISE_GATE      = 0.02 # RMS threshold below which a packet is muted (0 = off)
# -----------------------

packet_queue: collections.deque = collections.deque()
queue_lock = threading.Lock()
leftover: np.ndarray = np.zeros(0, dtype=np.float32)


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
        if NOISE_GATE > 0 and np.sqrt(np.mean(audio ** 2)) < NOISE_GATE:
            audio = np.zeros(SAMPLES_PER_PKT, dtype=np.float32)
        with queue_lock:
            if len(packet_queue) >= MAX_QUEUE_LEN:
                packet_queue.popleft()  # discard oldest to prevent growing lag
            packet_queue.append(audio)


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
                break  # underrun — remaining output stays as silence
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP audio on port {UDP_PORT}...")

    recv_thread = threading.Thread(target=receive_loop, args=(sock,), daemon=True)
    recv_thread.start()

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
