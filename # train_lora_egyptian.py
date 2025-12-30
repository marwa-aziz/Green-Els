# train_lora_egyptian.py
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import LoraLoaderMixin
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch, os

model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "./lora_egyptian"

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Apply LoRA mixin
LoraLoaderMixin.enable_lora(pipe.unet)
LoraLoaderMixin.enable_lora(pipe.text_encoder)

# Simple dataset loader (local images)
dataset = load_dataset("imagefolder", data_dir="egyptian_style", split="train")

def collate(examples):
    images = [e["image"].convert("RGB").resize((512,512)) for e in examples]
    return {"pixel_values": pipe.feature_extractor(images, return_tensors="pt").pixel_values}

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)

for epoch in range(3):  # small demo run
    for batch in loader:
        optimizer.zero_grad()
        loss = pipe.unet(batch["pixel_values"]).loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

pipe.save_pretrained(output_dir)
print("LoRA saved to", output_dir)
