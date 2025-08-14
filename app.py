import os
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device="cuda",
    skip_safety_check=True,
)
# AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)


def process_single_request(person_image, garment_image, garment_type):
    try:
        # Load images
        person_img = person_image.convert("RGB")
        cloth_img = garment_image.convert("RGB")

        # Resize images
        person_img = resize_and_crop(person_img, (768, 1024))
        cloth_img = resize_and_padding(cloth_img, (768, 1024))

        # Generate mask for specified garment type
        mask = automasker(person_img, garment_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        # Process with pipeline
        result = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )[0]

        return result

    except Exception as e:
        print(f"Error processing file-based request: {e}")
        return None


if __name__ == "__main__":
    temp = process_single_request(
        Image.open("person_image.png"),
        Image.open("cloth_image.png"),
        "overall",
    )

    if temp:
        temp.save("output_image.png")
        print("Image processed and saved as output_image.png")
