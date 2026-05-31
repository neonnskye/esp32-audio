import argparse
import os
import wave

import numpy as np


def convert(input_file, output_file=None, var_name=None, rate=8000, stereo=False):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".h"
    if var_name is None:
        var_name = (
            os.path.splitext(os.path.basename(input_file))[0]
            .replace("-", "_")
            .replace(" ", "_")
        )

    with wave.open(input_file, "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    print(f"  Channels   : {channels}")
    print(f"  Sample rate: {framerate} Hz")
    print(f"  Bit depth  : {sampwidth * 8} bit")
    print(f"  Frames     : {n_frames}")

    # Convert to 16-bit if needed
    samples = np.frombuffer(raw, dtype=np.int16 if sampwidth == 2 else np.int8)
    if sampwidth == 1:  # 8-bit unsigned → 16-bit signed
        samples = (samples.astype(np.int16) - 128) * 256

    if stereo:
        # Keep or produce interleaved stereo
        if channels == 1:
            # Mono → stereo: duplicate each sample: [S0, S1, ...] → [S0, S0, S1, S1, ...]
            samples = np.repeat(samples, 2)
        # If channels == 2 already, samples is already interleaved L/R — leave as-is
    else:
        # Mix down to mono
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample if needed
    if framerate != rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(framerate, rate)
        up = rate // g
        down = framerate // g

        if stereo:
            # Reshape to (n_frames, 2) and resample along axis 0 (frame axis)
            orig_frames = len(samples) // 2
            samples_2d = samples.reshape(orig_frames, 2)
            samples = (
                resample_poly(samples_2d, up, down, axis=0).astype(np.int16).ravel()
            )
        else:
            samples = resample_poly(samples, up, down).astype(np.int16)

        framerate = rate
        print(f"  Resampled to {rate} Hz")

    data = samples.tobytes()
    n = len(data)

    mode_str = "stereo" if stereo else "mono"

    with open(output_file, "w") as f:
        f.write(f"// Auto-generated from {os.path.basename(input_file)}\n")
        f.write(f"// {framerate} Hz, 16-bit, {mode_str}\n\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const uint32_t {var_name}_sample_rate = {framerate};\n")
        f.write(f"const uint32_t {var_name}_length = {n};\n\n")
        f.write(f"const int16_t {var_name}[] = {{\n  ")
        for i, s in enumerate(np.frombuffer(data, dtype=np.int16)):
            f.write(f"{s},")
            if (i + 1) % 16 == 0:
                f.write("\n  ")
        f.write("\n};\n")

    print(f"  Written to : {output_file}  ({n} bytes, {n // 2} samples, {mode_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAV file to C header")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument(
        "output", nargs="?", default=None, help="Output .h file (optional)"
    )
    parser.add_argument(
        "var_name", nargs="?", default=None, help="Variable name prefix (optional)"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=8000,
        help="Target sample rate in Hz (default: 8000)",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Output stereo (interleaved L/R) instead of mono",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.var_name, rate=args.rate, stereo=args.stereo)
