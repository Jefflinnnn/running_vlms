import json
import argparse
from data.hf_loader import hf_login
from data.dataset import CheXpertDataset
from models.medgemma4b import MedGemma4B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--subset",      type=str, default="findings")
    parser.add_argument("--split",       type=str, default="valid")
    parser.add_argument("--prompt",      type=str, default="Generate a radiology report for this chest X-ray.")
    args = parser.parse_args()

    hf_login()
    dataset = CheXpertDataset(subset=args.subset, split=args.split)

    model = MedGemma4B()
    model.load_model()

    results = []
    for idx in range(args.num_samples):
        image, _ = dataset[idx]
        try:
            output = model.generate(image, args.prompt)
        except Exception as e:
            output = f"ERROR: {str(e)}"
        results.append({"idx": idx, "output": output})
        print(f"  [medgemma4b] Sample {idx} done.", flush=True)

    # Print JSON to stdout — captured by run_inference.py
    print(json.dumps(results))


if __name__ == "__main__":
    main()