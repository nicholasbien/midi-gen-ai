"""
Generate v1 (GPT-2 finetuned) continuations on a prompt set, for v1 vs v2 eval.

Loads the v1 model from HuggingFace and runs it locally (CPU/MPS). Slower than
the Modal serving path but self-contained and reproducible.

Usage:
    python -m v2.eval_v1_generate \\
        --prompts preselected_midi/ --out evals/v1_samples \\
        --temperature 1.0 --top-k 10 --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from note_seq import midi_io
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

from convert import condense_note_sequence, expand_condensed_sequence


MODEL_NAME = "nicholasbien/gpt2_finetuned-lmd_full"


def load_v1():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    if device == "mps":
        model.half()
    return tokenizer, model, device


def encode_prompt(midi_path: Path) -> str:
    ns = midi_io.midi_file_to_note_sequence(str(midi_path))
    return condense_note_sequence(ns)


def decode_to_midi(text: str, out_path: Path) -> None:
    ns = expand_condensed_sequence(text)
    midi_io.note_sequence_to_midi_file(ns, str(out_path))


@torch.no_grad()
def generate_one(model, tokenizer, device, text: str,
                 temperature: float, top_k: int, max_new_tokens: int) -> str:
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=64,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, type=Path,
                   help="dir of prompt MIDI files (or single .mid)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.prompts.is_file():
        files = [args.prompts]
    else:
        files = sorted(args.prompts.glob("*.mid"))
        if args.limit:
            files = files[: args.limit]

    print(f"loading v1 model {MODEL_NAME}...")
    tokenizer, model, device = load_v1()
    print(f"  device={device}")

    for src in files:
        try:
            t0 = time.time()
            prompt_text = encode_prompt(src)
            out_text = generate_one(model, tokenizer, device, prompt_text,
                                    args.temperature, args.top_k, args.max_new_tokens)
            out_path = args.out / f"{src.stem}__t{args.temperature}.mid"
            decode_to_midi(out_text, out_path)
            print(f"  {src.name}  -> {out_path.name}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {src.name}  FAILED: {e}")


if __name__ == "__main__":
    main()
