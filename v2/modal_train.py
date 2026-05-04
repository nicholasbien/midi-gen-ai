"""
Modal training: train a v2 model on Modal-hosted H100 / A100.

Cost reference (Modal pricing as of 2026-05):
    H100  ~$3.99/hr  — fastest, best for any model >50M
    A100-80GB  ~$2.50/hr  — cheaper, ~3-4x slower than H100 on bf16
    A10G  ~$1.10/hr  — only worth it for tiny models

Volumes:
    openmusenet2-v2-corpus  — read-only training data (shards + tokenizer + manifest)
    openmusenet2-v2-runs    — checkpoints + train logs

Upload corpus once (from wherever the shards live, e.g. Lambda):
    modal volume put openmusenet2-v2-corpus /home/ubuntu/data/v2_corpus_full /

Launch training:
    modal run v2/modal_train.py --size medium --max-steps 15000 --gpu H100

Pull a checkpoint back:
    modal volume get openmusenet2-v2-runs <run-name>/ckpt_final.pt ./
"""

from __future__ import annotations

import modal
from modal import Image, Volume


CORPUS_VOLUME_NAME = "openmusenet2-v2-corpus"
RUNS_VOLUME_NAME = "openmusenet2-v2-runs"

app = modal.App("openmusenet2-train")

corpus_volume = Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=True)
runs_volume = Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("torch", "miditok", "symusic", "numpy", "tqdm")
    .add_local_python_source("v2")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={
        "/corpus": corpus_volume,
        "/runs": runs_volume,
    },
)
def train(
    size: str = "medium",
    max_steps: int = 15000,
    batch_size: int = 16,
    grad_accum: int = 4,
    block_size: int = 1024,
    warmup_steps: int = 200,
    eval_interval: int = 500,
    save_interval: int = 1000,
    lr: float = 3e-4,
    run_name: str | None = None,
) -> dict:
    import os
    from datetime import datetime
    from pathlib import Path

    from v2.train_v2 import train as train_fn, TrainConfig

    if not run_name:
        run_name = f"{size}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(f"/runs/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[modal-train] run_name={run_name}")
    print(f"[modal-train] corpus contents:")
    os.system("ls -la /corpus/ /corpus/shards/ 2>/dev/null | head -20")

    # corpus was uploaded as a subdir; point at the inner data root
    data_dir = Path("/corpus/v2_corpus_full")
    if not (data_dir / "shards").exists():
        # fallback: maybe it was uploaded at root
        data_dir = Path("/corpus")
    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        size=size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        block_size=block_size,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
        save_interval=save_interval,
        lr=lr,
        dtype="bfloat16",
    )
    train_fn(cfg)

    runs_volume.commit()  # flush checkpoints to volume
    return {"run_name": run_name, "out_dir": str(out_dir)}


@app.local_entrypoint()
def main(size: str = "medium", max_steps: int = 15000, batch_size: int = 16):
    """Local entrypoint — invoke training and print result."""
    result = train.remote(size=size, max_steps=max_steps, batch_size=batch_size)
    print(f"\n[done] {result}")
    print(f"\nRetrieve checkpoint with:")
    print(f"  modal volume get {RUNS_VOLUME_NAME} {result['run_name']}/ckpt_final.pt ./runs/")
