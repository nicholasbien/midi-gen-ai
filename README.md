# MIDI Gen AI

Fine-tuning transformers on text encodings of MIDI files, generating new music with the resulting model.

Demo at https://nicholasbien.com/midi.

The repo ships **two model generations side by side**: v1 (the original GPT-2 fine-tune) and v2 (custom transformers trained from scratch with event-based tokenization). The v2 line is what's currently serving https://nicholasbien.com/midi.

## Available checkpoints

| Tag | Params | Training data | Final loss | Best temp | HF |
|---|---|---|---|---|---|
| `v1` | 124M (GPT-2) | Lakh full | — | t=1.2, top_k=10 | [nicholasbien/gpt2_finetuned-lmd_full](https://huggingface.co/nicholasbien/gpt2_finetuned-lmd_full) |
| `v2-pilot` | 25M | Lakh + curated (~158k files, 2.5B tokens) | 0.97 | t=1.0, top_k=50 | (local-only) |
| `v2-prod` | 25M | + LAMD (~408k files, ~8B tokens) | 0.93 | t=1.0, top_k=50 | (local-only) |
| **`v2-100m`** ← live | **113M** | + LAMD (~408k files) | **0.71** | **t=1.2, top_k=50** | [nicholasbien/midigenai/v2-100m](https://huggingface.co/nicholasbien/midigenai/tree/main/v2-100m) |

---

## v1 → v2 at a glance

| | v1 | v2 (current: 113M) |
|---|---|---|
| Base model | GPT-2 (HuggingFace fine-tune) | Custom GPT-style transformer (from scratch) |
| Parameters | 124M | **113M** |
| Vocabulary | GPT-2 BPE (50,257) | Custom event vocab (**641**) |
| Tokenization | BPE on `pitch_start_dur_vel` text | [MidiTok MIDILike](https://github.com/Natooz/MidiTok) (event-based) |
| Tokens per note | ~6–10 | **~4** |
| Positional encoding | Learned absolute | RoPE (extrapolates beyond training length) |
| Activation / norm | GELU + LayerNorm | SwiGLU + RMSNorm |
| Attention | HF default | FlashAttention-2 via SDPA |
| Tempo handling | Encoded in text | Stripped at training, re-applied at decode (tempo-invariant learning) |
| Training data | Lakh full | Lakh + MAESTRO + POP909 + GiantMIDI + LAMD, deduped |
| Training infra | manual | one-shot Lambda Labs bootstrap (`v2/setup_lambda.sh`) |
| Eval | informal | quantitative metrics + A/B grading UI |

### Quantitative comparison (n=5 prompts, t=1.0)

Generated continuations from the same 5 prompt MIDIs, measured on the output:

| Metric | v1 | v2 (25M pilot) | Δ |
|---|---|---|---|
| Notes generated | 78 | 193 | +148% |
| Pitch class entropy | 1.91 | 2.63 | +37% |
| Pitch range (semitones) | 28 | 39 | +39% |
| Polyphony rate | 2.68 | 3.03 | +13% |
| **Repetition rate** (4-gram) | 0.48 | **0.19** | **−60%** |
| Inter-onset interval entropy | 1.51 | 2.26 | +50% |
| Scale consistency | 0.98 | 0.94 | −4% |

**Net:** v2 produces meaningfully more music with much less repetition, broader expressive range, and more rhythmic variety, while remaining tonally coherent.

### Inference latency (Mac)

| | tok/s | notes/s |
|---|---|---|
| v1 (124M) on CPU | 69 | ~9 |
| v1 (124M) on MPS | 93 | ~12 |
| v2 (25M) on CPU | **278** | **~70** |
| v2 (25M) on MPS | 172 | ~43 |

v2 is **~8x faster than v1 in actual notes/sec on CPU** (smaller model + 2x more efficient tokenization). MPS hurts v2 because per-kernel dispatch overhead dominates at this size — set `OMN_USE_MPS=1` to override.

---

## Repo layout

```
midi-gen-ai/
├─ convert.py            # v1: MIDI ↔ condensed text encoding
├─ dataset.py            # v1: HuggingFace dataset prep
├─ train.py              # v1: GPT-2 finetune
├─ generate.py           # v1: Modal-backed inference
│
└─ v2/
   ├─ tokenizer_v2.py        # MidiTok wrapper (event-based MIDILike)
   ├─ model_v2.py            # custom transformer: RoPE + SwiGLU + RMSNorm + SDPA
   ├─ train_v2.py            # training loop (sliding-window, AdamW, cosine LR, bf16)
   ├─ generate_v2.py         # streaming inference with KV cache + tempo plumbing
   ├─ eval_v2.py             # quantitative metrics
   ├─ eval_v1_generate.py    # generate v1 samples for v1 vs v2 comparison
   ├─ grade_app.py           # A/B grading frontend (Flask + html-midi-player)
   ├─ templates/grade.html
   ├─ data/
   │  ├─ download.py         # dataset fetchers (Lakh, MAESTRO, POP909, GiantMIDI, LAMD)
   │  ├─ clean.py            # quality filter + content-hash dedup
   │  ├─ build_dataset.py    # parallel tokenize + shard
   │  └─ augment.py          # pitch/velocity augmentation
   ├─ lambda_provision.py    # Lambda Cloud API CLI
   ├─ setup_lambda.sh        # one-shot Lambda training bootstrap
   └─ run_local_pilot.sh     # local pilot run on Mac CPU/MPS
```

---

## v2 quickstart

### Use a published model from anywhere (`pip install`)

The latest trained checkpoint is published at
[huggingface.co/nicholasbien/midigenai](https://huggingface.co/nicholasbien/midigenai).
Install the package and pull the model from there:

```bash
pip install git+https://github.com/nicholasbien/midi-gen-ai
```

```python
from midigenai import load_v2_from_hub, list_hub_versions

list_hub_versions()                       # ['v2-pilot', ...]
gen = load_v2_from_hub()                  # default version
gen = load_v2_from_hub(version="v2")      # pin to a specific subfolder

prompt = gen.encode_midi_file("seed.mid")
gen.generate_to_midi(prompt, "out.mid", tempo_bpm=80, max_new_tokens=512)
```

Files are cached at `~/.cache/huggingface/` after the first call. The default
version comes from `MIDIGENAI_VERSION` env var, then falls back to
`DEFAULT_VERSION` in `v2/hub.py`. To change the default for a shell session:

```bash
export MIDIGENAI_VERSION=v2
```

#### Publishing a new model

The HF repo is laid out as one subfolder per trained checkpoint:

```
nicholasbien/midigenai/
├── v2-pilot/        ← current default
│   ├── ckpt_final.pt
│   └── tokenizer.json
└── v2/              ← future
    └── ...
```

To release a new version, upload `ckpt_final.pt` + `tokenizer.json` to a new
subfolder. `list_hub_versions()` auto-discovers it. Either bump
`DEFAULT_VERSION` in `v2/hub.py` for a permanent default change, or have
callers set `MIDIGENAI_VERSION` themselves.

### Generate from a prompt (local checkpoint)

```bash
python -m v2.generate_v2 \
  --checkpoint runs/pilot/ckpt_final.pt \
  --tokenizer  runs/pilot/tokenizer.json \
  --input-midi your_prompt.mid \
  --output-midi continuation.mid \
  --temperature 1.0 --top-k 50
```

Output respects the input MIDI's tempo (auto-detected). Pass `--tempo-bpm 90` to override.

### A/B grading frontend

Compare two model checkpoints (or v1 vs v2) on paired prompts:

```bash
python -m v2.grade_app \
  --dir-a evals/v1_samples --label-a v1 \
  --dir-b evals/v2_samples --label-b v2 \
  --responses evals/grading_responses.csv
# open http://localhost:7777
```

Sides are randomized per pair, model labels hidden until you submit. Preferences are written to a CSV that's compatible with reward-model training.

### Quantitative eval

```bash
python -m v2.eval_v2 \
  evals/v1_samples evals/v2_samples \
  --label-a v1 --label-b v2 \
  --out evals/v1_vs_v2.json
```

Prints a side-by-side table of seven metrics (pitch class entropy, scale consistency, polyphony, note density, pitch range, repetition rate, inter-onset interval entropy).

### Train v2 on Lambda Labs

```bash
# 1. Generate a Lambda Cloud API key at https://cloud.lambdalabs.com/api-keys
echo "LAMBDA_API_KEY=secret_..." > ~/.lambda_env
chmod 600 ~/.lambda_env

# 2. Provision a single GPU instance (A10 is enough for 25M pilot)
python -m v2.lambda_provision list-types
python -m v2.lambda_provision launch \
  --instance-type gpu_1x_a10 --region us-west-1 \
  --ssh-key your-key-name --name midigenai-train

# 3. Wait for it to come online (prints SSH info)
python -m v2.lambda_provision wait <instance-id>

# 4. SSH in and run the bootstrap (downloads data, cleans, tokenizes, trains in tmux)
ssh ubuntu@<ip>
git clone https://github.com/nicholasbien/midi-gen-ai.git
cd midi-gen-ai
SIZE=pilot MAX_STEPS=5000 bash v2/setup_lambda.sh
```

The pilot 25M model trains in ~46 minutes on an A10 (~$1.50 in compute), reaching loss 0.97 from initial 6.55.

### Train v2 locally

```bash
bash v2/run_local_pilot.sh    # auto-downloads, cleans, tokenizes, trains on Mac
```

Slower but free.

---

## v2 architecture

### Tokenization

[MidiTok](https://github.com/Natooz/MidiTok) MIDILike — an event-based tokenizer descended from the original Music Transformer (Huang et al. 2018):

```
NoteOn_<pitch>     one token per pitch
Velocity_<bin>     32 bins
NoteOff_<pitch>
TimeShift_<n>      log-binned milliseconds
Program_<n>        instrument changes
Rest_<...>         silence
```

Why MIDILike over REMI: REMI's bar grid is rigid for live jamming (user input isn't quantized). MIDILike preserves microtiming.

**Tempo is stripped at training time** (`use_tempos=False`) and **re-applied at decode time**. This gives tempo-invariant learning (the same musical phrase at 80 BPM and 120 BPM tokenizes identically) and lets us match the user's tempo on output.

Total vocab: **641 tokens** (vs v1's 50,257 GPT-2 BPE).

### Model

| | Pilot | Production |
|---|---|---|
| d_model | 512 | 1024 |
| n_layers | 6 | 12 |
| n_heads | 8 | 16 |
| d_ff | 2048 | 4096 |
| Context | 1024 | 2048 |
| Params | **25.5M** | 202M |

All configs share: RoPE positional embeddings, SwiGLU feedforward, RMSNorm, FlashAttention-2 via `torch.nn.functional.scaled_dot_product_attention`, tied embedding/output projection.

The pilot config (25M) is the recommended starting point. It trains in <1 hour on one A10 and is competitive with v1's 124M model on every musical metric measured.

### Data pipeline

- **Download** (`v2/data/download.py`): fetches Lakh, MAESTRO, POP909, GiantMIDI, LAMD from their respective hosts.
- **Clean** (`v2/data/clean.py`): drops unparseable / too-short / too-long files. **Content-hash dedup** (SHA1 over `(pitch, start, duration)` tuples) catches exact duplicates across datasets — essential since Lakh / LAMD overlap heavily.
- **Build** (`v2/data/build_dataset.py`): parallel tokenization (all CPU cores), packs into uint16 numpy shards (50M tokens each).
- **Augment** (`v2/data/augment.py`): pitch shift ±6 semitones, velocity bin jitter, applied at training time.

Combined corpus after dedup: ~158k MIDIs from Lakh + curated; ~500k+ with LAMD added. ~2.5B training tokens (pilot), ~8B+ with LAMD.

---

## v1 (legacy)

The original GPT-2 system is still available — see `convert.py`, `dataset.py`, `train.py`, `generate.py` and the [v1 design notes](#detailed-design-and-next-steps-v1) for context. v1 is what's deployed at https://nicholasbien.com/midi today.

### **convert.py**

The `convert.py` script is a command-line utility that can encode MIDI files into a condensed text format or decode them back. It processes directories recursively.

```bash
python convert.py encode /path/to/input/folder /path/to/output/folder
python convert.py decode /path/to/input/folder /path/to/output/folder
```

### **dataset.py**

Reads text files from a directory, builds a HuggingFace dataset, splits train/val, pushes to the Hub. Defaults to the `lmd_full_txt` directory; configurable via `TXT_DIRECTORY`.

### **train.py**

Fine-tunes GPT-2 on the text dataset using HuggingFace `transformers`. Configurable via `DATASET_NAME`, `TOKENIZED_DATASET_NAME`, `TOKENIZER_NAME`, `BASE_MODEL_NAME`, `MODEL_NAME`.

### **generate.py**

Modal-backed inference (`gpu="any"`). Streams text completions and prints them. Configurable via `TOKENIZER_NAME`, `MODEL_NAME`, `text`, `num_return_sequences`, `temperature`, `top_k`.

---

## Setup

```bash
git clone https://github.com/nicholasbien/midi-gen-ai.git
cd midi-gen-ai
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Optional (depending on what you're running):

| For | Extra setup |
|---|---|
| v1 inference / training | `huggingface-cli login` |
| v1 Modal serving | `modal token new` |
| v2 Lambda training | API key in `~/.lambda_env` |
| HF dataset downloads | `HF_TOKEN` env var (avoids rate limits) |

## Dependencies

See `requirements.txt`. Highlights:

- **v1**: `transformers`, `datasets`, `accelerate`, `trl`, `note_seq`
- **v2**: `miditok`, `symusic`, `torch`, `einops`, `flask` (for grading UI), `tqdm`

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Detailed design and next steps (v1)

The v1 design notes from the original README are preserved below for reference. Items resolved by v2 are annotated.

### Dataset
- There are lots of other large MIDI datasets out there. Add these to the training data. In experiments with finetuning on the LMD-matched (45,129 MIDI files) vs. the LMD-full (176,581 MIDI files) it is pretty clear that data is the most important lever for improving generations. *(Resolved in v2: corpus expanded to ~500k MIDIs across 5 sources with content-hash dedup.)*
- Currently only the first max_sequence_length tokens of each MIDI file are parsed. Chunking MIDI files up into max_sequence_length-size chunks would increase the training data volume significantly (~100x). *(Resolved in v2: sliding-window training over packed shards.)*
- Support multiple tracks. Many of the MIDI files in the dataset have multiple tracks. Currently only the first track in the file is processed. *(Resolved in v2: `use_programs=True` interleaves multi-instrument tokens into a single autoregressive stream.)*
- There is probably some leakage due to near-duplicates MIDI files in both the training and validation sets. *(Partially resolved in v2: exact-content dedup via SHA1 over `(pitch, start, duration)` tuples. Near-duplicate detection is still future work.)*

### Encoding
- There is a minor mistake in the encoding: note velocity is encoded as a float instead of an int. *(Resolved in v2: 32 velocity bins as discrete tokens.)*
- Timing is encoded in seconds. The model learns timing pretty well, but it is at times imprecise. Encoding the start of each measure or beat could help. *(Resolved in v2: timing in beat-relative ticks via TimeShift tokens, tempo stripped at training time and re-applied at decode.)*
- A preferred method of encoding timing might be to use one beat as the timing unit instead of one second and additionally encode the BPM. *(Resolved in v2: exactly this scheme.)*

### Tokenizer
- v1 uses the default GPT-2 tokenizer. Experiments with custom tokenizers gave worse results, likely due to small dataset size. *(Resolved in v2: 641-token event-based MIDILike vocab via MidiTok, ~2x more efficient than v1's BPE.)*
- GPT-3+ tokenizers have improvements to numerical encoding that would help with timing. *(Sidestepped in v2: timing is no longer represented as floating-point text, so tokenizer numerical handling doesn't matter.)*

### Model
- v1 uses GPT-2 small (~117M parameters) because it's open source and easy to train/inference on 1 modest GPU. Llama-7B was too cost-prohibitive. *(v2 trains a 25M custom model from scratch — smaller, faster, better musical quality, ~$1.50 to train on Lambda.)*

### Finetuning
- Finetuning is better than just prompting because input/output is not similar to natural language. The base GPT-2 model gives funny outputs like:
    - ```54.19_04.15_99.8 53_19.44_05.28_94.16 ... Now, let's take a look at the numbers ...```
    - ```26.42_97.23_84.64 ... Quotes are not sourced from all markets...```
- v1 finetunes from GPT-2; v2 trains from scratch since the music vocab is small enough that we don't need a strong language prior. The text prior actively wastes capacity in v1.

### Generation
- Generated MIDI = user prompt + model response. Response is usually after the prompt but sometimes is overlaid on top of the prompt.
- Sequence length is limited by GPT-2 model size (1024 tokens) — long enough for 1-2 minutes of single-line music, but only a few measures of chords. *(Resolved in v2: 2048 context, with RoPE that extrapolates beyond training length.)*
- GPU inference via Modal: ~10 seconds end-to-end on v1. *(v2 is ~8x faster in notes/sec, see latency table above.)*
- Generation parameters
    - Temperature: ~1.0–1.2 sweet spot for v1 melodies; ~1.0 with top-k 50 is the v2 sweet spot.
    - Top_k: ~10 for v1 (smaller vocab made larger top-k unstable). v2 handles top-k 50 fine.

### RLHF
- In the web interface, 2 responses are generated for each user prompt. The user has the option to pick which generation they prefer. The generations and user preferences are stored and can be used later to train a reward model as in equation 1 from "Training language models to follow instructions with human feedback" (https://arxiv.org/abs/2203.02155). *(v2 keeps this loop and adds a dedicated A/B grading frontend at `v2/grade_app.py` that randomizes left/right and writes a CSV compatible with reward-model training.)*
