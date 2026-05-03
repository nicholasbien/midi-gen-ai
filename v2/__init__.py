"""
v2 — custom 25M transformer for MIDI continuation.

Public surface:
- `V2Generator`           load a checkpoint and stream / batch generate
- `load_v2_from_hub`      one-liner: pull a versioned model from HuggingFace
                          and return a generator
- `list_hub_versions`     list available model subfolders on the HF repo
"""

from v2.generate_v2 import V2Generator, Note
from v2.hub import (
    load_v2_from_hub,
    download_v2_files,
    list_hub_versions,
    DEFAULT_REPO,
    DEFAULT_VERSION,
)

__all__ = [
    "V2Generator",
    "Note",
    "load_v2_from_hub",
    "download_v2_files",
    "list_hub_versions",
    "DEFAULT_REPO",
    "DEFAULT_VERSION",
]
