#!/bin/bash

# split_wav.sh
# Splits all WAV files in a directory into 1-second clips, placed in a clips/ subfolder.
# Usage: ./split_wav.sh [directory]
#   directory: optional, defaults to current directory

RAW_DIR="${1:-.}"

# Convert Windows-style paths (C:\... or C:/...) to WSL Linux paths (/mnt/c/...)
if echo "$RAW_DIR" | grep -qEi '^[a-zA-Z]:[/\\]'; then
    DRIVE_LETTER=$(echo "$RAW_DIR" | cut -c1 | tr '[:upper:]' '[:lower:]')
    # Strip the drive letter + colon + leading slash/backslash, then normalize slashes
    WIN_PATH=$(echo "$RAW_DIR" | sed 's/^[a-zA-Z]:[\/\\]//' | sed 's/\\/\//g')
    TARGET_DIR="/mnt/$DRIVE_LETTER/$WIN_PATH"
else
    # Normalize any remaining backslashes to forward slashes (for mixed-style input)
    TARGET_DIR=$(echo "$RAW_DIR" | sed 's/\\/\//g')
fi
CLIPS_DIR="$TARGET_DIR/clips"

mkdir -p "$CLIPS_DIR"

shopt -s nullglob
WAV_FILES=("$TARGET_DIR"/*.wav)

if [ ${#WAV_FILES[@]} -eq 0 ]; then
    echo "No WAV files found in '$TARGET_DIR'"
    exit 1
fi

echo "Found ${#WAV_FILES[@]} WAV file(s). Output goes to: $CLIPS_DIR"
echo "-------------------------------------------"

for f in "${WAV_FILES[@]}"; do
    filename=$(basename "$f")
    base="${filename%.wav}"

    echo ""
    echo "[1/4] Getting duration: $filename"
    duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
    echo "      Duration: ${duration}s"

    echo "[2/4] Padding to next whole second..."
    # Ceiling: if duration is already a whole number, keep it; otherwise round up
    rounded=$(echo "$duration" | awk '{print int($1) == $1 ? int($1) : int($1)+1}')
    echo "      Padding to: ${rounded}s"
    padded_file="$TARGET_DIR/${base}_padded.wav"
    ffmpeg -v error -i "$f" -af "apad=pad_dur=1" -t "$rounded" "$padded_file" -y

    echo "[3/4] Splitting into 1-second clips..."
    ffmpeg -v error -i "$padded_file" -f segment -segment_time 1 "$CLIPS_DIR/${base}.s%d.wav"
    rm "$padded_file"

    echo "[4/4] Fixing any short clips..."
    fixed=0
    for clip in "$CLIPS_DIR/${base}".s*.wav; do
        clip_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$clip")
        # Use awk to check if duration is less than 0.999s (handles floating point variations)
        is_short=$(echo "$clip_dur" | awk '{print ($1 < 0.999) ? "1" : "0"}')
        if [ "$is_short" = "1" ]; then
            fixed_file="${clip%.wav}_fixed.wav"
            ffmpeg -v error -i "$clip" -af "apad=pad_dur=1" -t 1 "$fixed_file" -y
            # NTFS-safe replace: delete original first, then move
            rm "$clip" && mv "$fixed_file" "$clip"
            ((fixed++))
        fi
    done
    echo "      Fixed $fixed short clip(s)."

    echo "Done: $filename"
done

echo ""
echo "-------------------------------------------"
echo "All files processed. Clips saved to: $CLIPS_DIR"