import json
import argparse
from data.hf_loader import hf_login
from data.dataset import CheXpertDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--subset",      type=str, default="findings")
    parser.add_argument("--split",       type=str, default="valid")
    args = parser.parse_args()

    hf_login()
    dataset = CheXpertDataset(subset=args.subset, split=args.split)
    ground_truths = [dataset[i][1] for i in range(args.num_samples)]

    # Print JSON to stdout — captured by run_inference.py
    print(json.dumps(ground_truths))


if __name__ == "__main__":
    main()