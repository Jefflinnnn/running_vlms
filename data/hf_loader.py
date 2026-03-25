import os
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from huggingface_hub import login

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────

def hf_login():
    """
    Authenticate with HuggingFace using the HF_TOKEN environment variable.
    
    Setup instructions:
        Windows PowerShell:
            [System.Environment]::SetEnvironmentVariable("HF_TOKEN", "your_token", "User")
        Mac/Linux:
            echo 'export HF_TOKEN="your_token"' >> ~/.zshrc && source ~/.zshrc
        Or use the CLI:
            huggingface-cli login
    """
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "See docstring for setup instructions."
        )
    login(token=token)

# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

DATASET_NAME = "X-iZhang/CheXpert-plus-RRG"

SUBSETS = {
    "findings": "findings_section",
    # Note: The 'impression' subset is actually the same as 'findings' in this dataset, but we'll keep the naming for clarity.
    # Sigh
    "impression": "impression_section",
}

SPLITS = ["valid"]

def load_chexpert(
    subset: str = "findings",
    split: str = "valid",
    cache_dir: str = "data/hf_cache"
) -> object:
    """
    Load a CheXpert+ subset from HuggingFace with local caching.

    Args:
        subset:    One of 'findings' or 'impression'.
        split:     One of 'valid'.
        cache_dir: Local directory to cache the dataset.

    Returns:
        HuggingFace Dataset object.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset '{subset}'. Choose from: {list(SUBSETS.keys())}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose from: {SPLITS}")

    print(f"Loading '{subset}' subset, '{split}' split...")
    ds = load_dataset(DATASET_NAME, name=SUBSETS[subset], split=split, cache_dir=cache_dir)
    print(f"Loaded {len(ds)} records with columns: {ds.column_names}")
    return ds

# ──────────────────────────────────────────────
# Image + Text Utilities
# ──────────────────────────────────────────────

def get_image(ds, idx: int = 0) -> Image.Image:
    """Extract a single image as RGB PIL from the dataset."""
    return ds[idx]["main_image"].convert("RGB")

def get_text(ds, idx: int = 0) -> str:
    """Extract the report text for a given image."""
    return ds[idx]["findings_section"]

def get_sample(ds, idx: int = 0) -> tuple:
    """Return (image, text) pair for a given index."""
    return get_image(ds, idx), get_text(ds, idx)

if __name__ == "__main__":
    hf_login()
    ds = load_chexpert(subset="findings", split="valid")
    print(f"Loaded {len(ds)} records with columns: {ds.column_names}")