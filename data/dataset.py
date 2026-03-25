from torch.utils.data import Dataset
from hf_loader import hf_login, load_chexpert, get_sample

class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for CheXpert+ findings or impression subset.

    Each item returns:
        image: PIL Image in grayscale (RGB mode)
        text:  Report text (findings or impression)
    """
    def __init__(self, subset: str = "findings", split: str = "valid"):
        self.ds = load_chexpert(subset=subset, split=split)
        self.subset = subset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple:
        return get_sample(self.ds, idx)


if __name__ == "__main__":
    hf_login()

    dataset = CheXpertDataset(subset="findings", split="valid")
    print(f"")
    print(f"Dataset size: {len(dataset)}")

    image, text = dataset[0]
    image.show()
    print(f"Image size: {image.size}, mode: {image.mode}")
    print(f"")
    print(f"Text:\n{text}")