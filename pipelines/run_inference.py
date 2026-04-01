import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

NUM_SAMPLES = 5
SUBSET      = "findings"
SPLIT       = "valid"
PROMPT      = "Generate a radiology report for this chest X-ray."
OUTPUT_DIR  = Path("outputs")

# Toggle: True = stop pipeline on first model failure, False = skip and continue
FAIL_FAST = False

# Venv Python executables
VENV_PYTHONS = {
    "medgemma4b": Path("venv/medgemma/Scripts/python.exe"),
    "medgemma27b": Path("venv/medgemma/Scripts/python.exe"),
    "chexone":    Path("venv/chexone/Scripts/python.exe"),
}

# Per-model inference scripts
MODEL_SCRIPTS = {
    "medgemma4b": Path("pipelines/inference_medgemma4b.py"),
    "medgemma27b": Path("pipelines/inference_medgemma27b.py"),
    "chexone":    Path("pipelines/inference_chexone.py"),
}

# Venv for loading ground truth (has datasets library installed)
GROUND_TRUTH_PYTHON = Path("venv/.venv-3.14/Scripts/python.exe")

# ──────────────────────────────────────────────
# Ground Truth Loader Script
# ──────────────────────────────────────────────

# Runs in GROUND_TRUTH_PYTHON venv, prints JSON list of ground truth texts to stdout
GROUND_TRUTH_SCRIPT = Path("pipelines/load_ground_truth.py")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def validate_config():
    """Check all venv paths and scripts exist before starting."""
    errors = []
    for model_name, python in VENV_PYTHONS.items():
        if not python.exists():
            errors.append(f"Venv not found for {model_name}: {python}")
    for model_name, script in MODEL_SCRIPTS.items():
        if not script.exists():
            errors.append(f"Script not found for {model_name}: {script}")
    if not GROUND_TRUTH_PYTHON.exists():
        errors.append(f"Ground truth venv not found: {GROUND_TRUTH_PYTHON}")
    if not GROUND_TRUTH_SCRIPT.exists():
        errors.append(f"Ground truth script not found: {GROUND_TRUTH_SCRIPT}")
    if errors:
        for e in errors:
            print(f"  [CONFIG ERROR] {e}")
        sys.exit(1)
    print("Config validated.")


def run_subprocess(name: str, python: Path, script: Path, extra_args: list) -> str | None:
    """
    Run a script as a subprocess using the given Python executable.
    Returns stdout string on success, None on failure.
    """
    cmd = [str(python), str(script)] + extra_args
    print(f"\n[{name}] Starting subprocess: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[{name}] FAILED with return code {result.returncode}")
        print(f"[{name}] stderr:\n{result.stderr}")
        return None

    return result.stdout


# ──────────────────────────────────────────────
# Ground Truth
# ──────────────────────────────────────────────

def load_ground_truth(num_samples: int) -> list:
    """Load ground truth texts by running load_ground_truth.py as a subprocess."""
    print("\nLoading ground truth...")
    extra_args = [
        "--num_samples", str(num_samples),
        "--subset", SUBSET,
        "--split", SPLIT,
    ]
    stdout = run_subprocess(
        "ground_truth", GROUND_TRUTH_PYTHON, GROUND_TRUTH_SCRIPT, extra_args
    )
    if stdout is None:
        print("ERROR: Could not load ground truth. Aborting.")
        sys.exit(1)
    try:
        ground_truths = json.loads(stdout)
        print(f"Loaded {len(ground_truths)} ground truth records.")
        return ground_truths
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse ground truth output:\n{stdout}")
        sys.exit(1)


# ──────────────────────────────────────────────
# Model Inference
# ──────────────────────────────────────────────

def run_model(model_name: str, num_samples: int) -> list | None:
    """
    Run a model inference script as a subprocess.
    Returns list of dicts with idx and output, or None on failure.
    """
    python = VENV_PYTHONS[model_name]
    script = MODEL_SCRIPTS[model_name]
    extra_args = [
        "--num_samples", str(num_samples),
        "--subset",      SUBSET,
        "--split",       SPLIT,
        "--prompt",      PROMPT,
    ]

    stdout = run_subprocess(model_name, python, script, extra_args)

    if stdout is None:
        return None

    try:
        results = json.loads(stdout)
        print(f"[{model_name}] Received {len(results)} results.")
        for r in results:
            print(f"  [{model_name}] Sample {r['idx']}: {str(r['output'])[:80]}...")
        return results
    except json.JSONDecodeError:
        print(f"[{model_name}] ERROR: Could not parse output:\n{stdout}")
        return None


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def run_pipeline(num_samples: int = NUM_SAMPLES):
    OUTPUT_DIR.mkdir(exist_ok=True)
    validate_config()

    # Load ground truth
    ground_truths = load_ground_truth(num_samples)

    # Run each model
    all_results = {}
    failed_models = []

    for model_name in MODEL_SCRIPTS:
        results = run_model(model_name, num_samples)

        if results is None:
            failed_models.append(model_name)
            if FAIL_FAST:
                print(f"\nFAIL_FAST=True — aborting pipeline after {model_name} failure.")
                sys.exit(1)
            else:
                print(f"[{model_name}] Skipping due to failure, continuing pipeline.")
                all_results[model_name] = [
                    {"idx": i, "output": "ERROR: subprocess failed"}
                    for i in range(num_samples)
                ]
        else:
            all_results[model_name] = results

    # Combine results by index
    print("\nCombining results...")
    combined = []
    for idx in range(num_samples):
        sample = {
            "idx":          idx,
            "ground_truth": ground_truths[idx],
        }
        for model_name in MODEL_SCRIPTS:
            sample[model_name] = all_results[model_name][idx]["output"]
        combined.append(sample)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"inference_{SUBSET}_{SPLIT}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Pipeline complete.")
    print(f"Samples:       {num_samples}")
    print(f"Models run:    {list(MODEL_SCRIPTS.keys())}")
    if failed_models:
        print(f"Failed models: {failed_models}")
    print(f"Output:        {output_path}")
    print(f"{'='*50}")

    # Print first sample as sanity check
    print("\nSample output [0]:")
    print(json.dumps(combined[0], indent=2))

if __name__ == "__main__":
    run_pipeline(num_samples=NUM_SAMPLES)