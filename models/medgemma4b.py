from models.medgemma import MedGemmaBase


class MedGemma4B(MedGemmaBase):
    """MedGemma 4B instruction-tuned variant."""
    MODEL_ID = "google/medgemma-1.5-4b-it"


if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()
    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    model = MedGemma4B()
    model.load_model()

    output = model.generate(image, "Generate a radiology report for this chest X-ray.")
    print(f"Ground truth:\n{findings_text}")
    print(f"\nMedGemma 4B output:\n{output}")