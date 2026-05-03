"""
Clean a directory of raw MIDI files: drop unparseable / too-small / duplicate files.

Outputs a list of paths that pass the filter, plus stats. Doesn't move files —
the next stage (build_dataset.py) reads paths from the manifest.

Filters:
- file opens with symusic (drops corrupt MIDIs)
- >= MIN_NOTES total notes
- <= MAX_DURATION_SECONDS (drops corrupt long-tail)
- exact-duplicate dedup via SHA1 of (pitch, start_tick, duration_tick) tuples
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm

MIN_NOTES = 32
MAX_DURATION_SECONDS = 60 * 30  # 30 min — anything longer is almost certainly broken
MIDI_EXTENSIONS = {".mid", ".midi"}


@dataclass
class FileStats:
    path: str
    n_notes: int
    n_tracks: int
    duration_seconds: float
    content_hash: str


def iter_midi_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in MIDI_EXTENSIONS and p.is_file():
            yield p


def hash_score(score) -> str:
    """Stable hash over the (pitch, start, duration) of every note across tracks."""
    h = hashlib.sha1()
    for track in score.tracks:
        for note in track.notes:
            h.update(int(note.pitch).to_bytes(2, "little", signed=False))
            h.update(int(note.start).to_bytes(8, "little", signed=False))
            h.update(int(note.duration).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def inspect(path: Path) -> FileStats | None:
    try:
        from symusic import Score
        score = Score(str(path))
    except Exception:
        return None

    try:
        # symusic exposes ticks_per_quarter and tempos; we approximate duration
        # by max end of any note converted via the first tempo.
        tpq = score.ticks_per_quarter
        tempos = score.tempos
        bpm = tempos[0].qpm if len(tempos) > 0 else 120.0
        seconds_per_tick = 60.0 / (bpm * tpq)

        n_notes = 0
        max_end_tick = 0
        for track in score.tracks:
            for note in track.notes:
                n_notes += 1
                end = note.start + note.duration
                if end > max_end_tick:
                    max_end_tick = end
        duration_seconds = max_end_tick * seconds_per_tick

        if n_notes < MIN_NOTES:
            return None
        if duration_seconds > MAX_DURATION_SECONDS:
            return None

        return FileStats(
            path=str(path),
            n_notes=n_notes,
            n_tracks=len(score.tracks),
            duration_seconds=duration_seconds,
            content_hash=hash_score(score),
        )
    except Exception:
        return None


def clean(input_root: Path, manifest_out: Path) -> dict:
    seen_hashes: set[str] = set()
    kept: list[FileStats] = []
    n_total = 0
    n_unparseable = 0
    n_too_small = 0
    n_duplicate = 0

    for path in tqdm(list(iter_midi_files(input_root)), desc="cleaning"):
        n_total += 1
        stats = inspect(path)
        if stats is None:
            n_unparseable += 1  # also covers "too small"; we don't distinguish
            continue
        if stats.content_hash in seen_hashes:
            n_duplicate += 1
            continue
        seen_hashes.add(stats.content_hash)
        kept.append(stats)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w") as f:
        for s in kept:
            f.write(json.dumps(asdict(s)) + "\n")

    summary = {
        "input_root": str(input_root),
        "manifest": str(manifest_out),
        "n_total": n_total,
        "n_kept": len(kept),
        "n_dropped_unparseable_or_small": n_unparseable + n_too_small,
        "n_dropped_duplicate": n_duplicate,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="directory of raw MIDI files (recursive)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="output JSONL manifest of kept files")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if manifest already exists")
    args = parser.parse_args()
    if args.manifest.exists() and not args.force:
        n = sum(1 for _ in args.manifest.open())
        print(f"[skip] manifest {args.manifest} already exists with {n} entries; use --force to re-run")
    else:
        clean(args.input, args.manifest)
