"""
HuggingFace download helper. Loads a `V2Generator` straight from
[`nicholasbien/midigenai`](https://huggingface.co/nicholasbien/midigenai) (or
any compatible repo) so callers don't have to manage checkpoint paths.

```python
from v2.hub import load_v2_from_hub
gen = load_v2_from_hub()                     # latest default version
gen = load_v2_from_hub(version="v2-pilot")   # pin to a specific subfolder
```

Files are cached at `~/.cache/huggingface/` after the first download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_REPO = "nicholasbien/midigenai"
DEFAULT_VERSION = "v2-pilot"


def load_v2_from_hub(
    version: str = DEFAULT_VERSION,
    repo_id: str = DEFAULT_REPO,
    revision: Optional[str] = None,
    **generator_kwargs,
):
    """Download checkpoint + tokenizer from HF and return a ready V2Generator."""
    from huggingface_hub import hf_hub_download
    from v2.generate_v2 import V2Generator

    ckpt = hf_hub_download(repo_id, "ckpt_final.pt", subfolder=version, revision=revision)
    tok  = hf_hub_download(repo_id, "tokenizer.json", subfolder=version, revision=revision)
    return V2Generator(checkpoint_path=ckpt, tokenizer_path=tok, **generator_kwargs)


def download_v2_files(
    version: str = DEFAULT_VERSION,
    repo_id: str = DEFAULT_REPO,
    revision: Optional[str] = None,
) -> tuple[Path, Path]:
    """Download just the files; return (ckpt_path, tokenizer_path)."""
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(repo_id, "ckpt_final.pt", subfolder=version, revision=revision)
    tok  = hf_hub_download(repo_id, "tokenizer.json", subfolder=version, revision=revision)
    return Path(ckpt), Path(tok)
