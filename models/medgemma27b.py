from models.medgemma import MedGemmaBase


class MedGemma27B(MedGemmaBase):
    """
    MedGemma 27B instruction-tuned variant.
    Requires significantly more VRAM (~60GB+ for bfloat16).
    device_map='auto' will shard across available GPUs if needed.
    """
    MODEL_ID = "google/medgemma-27b-it"


if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()
    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    model = MedGemma27B()
    model.load_model()

    output = model.generate(image, "Generate a radiology report for this chest X-ray.")
    print(f"Ground truth:\n{findings_text}")
    print(f"\nMedGemma 27B output:\n{output}")