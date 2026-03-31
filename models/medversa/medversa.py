import sys
import os
import torch
from torch import cuda
from huggingface_hub import snapshot_download

repo_path = snapshot_download("hyzhou/MedVersa")
sys.path.insert(0, repo_path)
os.chdir(repo_path)

import utils as medversa_utils
medversa_utils.task_seg_2d    = lambda *args, **kwargs: None
medversa_utils.task_seg_3d    = lambda *args, **kwargs: None
medversa_utils.seg_2d_process = lambda *args, **kwargs: (None, None)
medversa_utils.seg_3d_process = lambda *args, **kwargs: (None, None)
medversa_utils.det_2d_process = lambda *args, **kwargs: None

from utils import *

device = 'cuda' if cuda.is_available() else 'cpu'
model_cls = registry.get_model_class('medomni')
model = model_cls.from_pretrained('hyzhou/MedVersa').to(device).eval()
print(f"embed_tokens device: {next(model.embed_tokens.parameters()).device}")
print(f"llama_model device:  {next(model.llama_model.parameters()).device}")
print(f"visual_encoder device: {next(model.visual_encoder_2d.parameters()).device}")

# ── Tokenizer diagnostic ──
tokenizer = model.llama_tokenizer
print(f"pad_token:          {tokenizer.pad_token!r}")
print(f"pad_token_id:       {tokenizer.pad_token_id}")
print(f"<ImageHere> id:     {tokenizer.convert_tokens_to_ids('<ImageHere>')}")
prefix = '###Human:' + '<img0>' + '<ImageHere>' * 9 + '</img0>'
tokens = tokenizer(prefix, return_tensors="pt")
print(f"32000 appears:      {(tokens.input_ids == 32000).sum().item()} times")

# ── Direct generate() call ──
image_tensor = medversa_utils.read_image(
    "./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png",
    "cxr", "report"
).to(device)
image_tensor = torch.cat([image_tensor])

with torch.autocast(device):
    with torch.no_grad():
        generated_image, seg_mask_2d, seg_mask_3d, output_text = medversa_utils.generate(
            model,
            ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
            image_tensor,
            "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea. PICC line placement.\nComparison: None.",
            "cxr",
            "report",
            1,        # num_imgs
            "How would you characterize the findings from <img0>?",
            1, True, 1, 0.9, 1, 1, 0.1,
            device,
        )
print(f"Direct generate() output: {output_text}")