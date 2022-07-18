import sys
from datetime import datetime
import os
import os.path

import ruclip
from rudalle import get_rudalle_model, get_vae, get_tokenizer, get_realesrgan
from rudalle.pipelines import generate_images, show, cherry_pick_by_ruclip, super_resolution
from rudalle.utils import seed_everything
import torch
from translatepy import Translate
print("Imports Complete")

class Args:
    def __init__(self):
        self.num_picturs = 1
        self.checkpoint_path = '/data/workspace/checkpoints'
        self.model_name = 'better_model'
        self.output_image_path = '/data/workspace/output_images'
        self.rudalle_cache_dir = '/data/workspace/rudalle'

def save_pil_images(pil_images, prompt_text) -> [str]:
    args = Args()
    out = []
    current_time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    for k in range(len(pil_images)):
        output_name = f"lg_{k}_{current_time}_{prompt_text}.png"
        out_file_path = os.path.join(args.output_image_path, output_name)
        pil_images[k].save(out_file_path)
    return out

model_generation_args = Args()

translation_engine = Translate()

# Prepare model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device, cache_dir=model_generation_args.rudalle_cache_dir)

vae = get_vae(dwt=True).to(device)

model_path = f"{model_generation_args.checkpoint_path}/{model_generation_args.model_name}_dalle_last.pt"
# model_path = os.path.join(model_generation_args.checkpoint_path, f"{model_generation_args.model_name}_dalle_last.pt")

model.load_state_dict(torch.load(model_path))

tokenizer = get_tokenizer()

realesrgan = get_realesrgan('x2', device=device)

clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)

clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)

print(":: Model Initialized")

seed_everything(42)

text_input = 'A happy couple'

print(f"Original Text: {text_input}")

text = translation_engine.translate(text_input, "ru").result
text = text_input
print(f"Translated Text: {text}")

pil_images = []
scores = []

for top_k, top_p, images_num in [
    (2048, 0.995, model_generation_args.num_picturs),
]:
    _pil_images, _scores = generate_images(text, tokenizer, model, vae, top_k=top_k, images_num=images_num, bs=8, top_p=top_p)
    pil_images += _pil_images
    scores += _scores

print(":: Image Generation Complete")

# show(pil_images)

print(":: Picking Top Images")
top_images, clip_scores = cherry_pick_by_ruclip(pil_images, text, clip_predictor, count=model_generation_args.num_picturs)

# show(top_images, model_generation_args.num_picturs)

print(":: Generating High Resolution")
sr_images = super_resolution(top_images, realesrgan)

# show(sr_images, model_generation_args.num_picturs)

out_paths = save_pil_images(sr_images, text_input.replace(' ', "_"))

print(f":: Images Saved To {out_paths}")