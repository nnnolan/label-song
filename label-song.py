import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

# Krumhansl-Kessler key profiles, starting from C
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def separate_vocals(path, out_dir):
    subprocess.run(
        [sys.executable, "-m", "demucs", "-d", "cpu", "--two-stems=vocals", "-n", "htdemucs", "-o", str(out_dir), str(path)],
        check=True,
    )
    return out_dir / "htdemucs" / path.stem / "vocals.wav"


def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best = None
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(12):
            score = np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]
            if best is None or score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def vocal_range(y, sr, min_prob):
    f0, _, prob = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    keep = ~np.isnan(f0) & (prob >= min_prob)
    if not keep.any():
        return None
    midi = librosa.hz_to_midi(f0[keep])
    low, tess_low, tess_high, high = np.percentile(midi, [5, 25, 75, 95]).round().astype(int)
    return {
        "low_note": int(low),
        "high_note": int(high),
        "tess_low": int(tess_low),
        "tess_high": int(tess_high),
        "confidence": round(float(keep.mean()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Label a song's key and vocal range.")
    parser.add_argument("audio", type=Path, help="mp3 or other audio file")
    parser.add_argument("--no-separate", action="store_true", help="skip vocal separation (input is already vocals)")
    parser.add_argument("--min-prob", type=float, default=0.8, help="minimum voicing probability per frame")
    parser.add_argument("--out", type=Path, default=Path("separated"), help="folder for separated stems")
    args = parser.parse_args()

    mix, sr = librosa.load(args.audio, sr=22050, mono=True)
    tonic, mode = estimate_key(mix, sr)

    vocals_path = args.audio if args.no_separate else separate_vocals(args.audio, args.out)
    vocals, _ = librosa.load(vocals_path, sr=sr, mono=True)
    result = vocal_range(vocals, sr, args.min_prob)

    output = {"file": args.audio.name, "key": tonic, "key_name": PITCH_CLASSES[tonic], "mode": mode}
    if result is None:
        output["range_source"] = "pending"
    else:
        output.update(result)
        output["range_source"] = "pipeline"
        for field in ("low_note", "high_note", "tess_low", "tess_high"):
            output[field + "_name"] = librosa.midi_to_note(result[field])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()