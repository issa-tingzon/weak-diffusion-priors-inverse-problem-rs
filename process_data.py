from pathlib import Path
from PIL import Image
from pathlib import Path
from torchvision.datasets import CelebA
from PIL import Image
import random
random.seed(7)

INPUT_DIRS = [
    Path("data/bedroom"),
    Path("data/church"),
    Path("data/celeba/")
]
OUTPUT_ROOT = Path("data/")
NUM_SAMPLES = 100
CROP_SIZE = 256

def main():
    # Download the Bedroom and Church datasets from LSUN
    # using the code in this repo: https://github.com/fyu/lsun

    for category in ["bedroom", "church"]:
        input_dir = Path(f"lsun/data/{category}")
        output_dir = Path(f"data/{category}")

        output_dir.mkdir(parents=True, exist_ok=True)
        images = list(input_dir.rglob("*.webp"))

        for i, image_path in enumerate(images):
            with Image.open(image_path) as img:
                img.convert("RGB").save(output_dir / f"{i:05d}.png")
        print(f"Saved to: {output_dir}")

    # Download CelebA
    CelebA(root="data", split="test", download=True)

    for input_dir in INPUT_DIRS:
        images = sorted(input_dir.rglob("*.jpg"))
        sampled_images = random.sample(images, NUM_SAMPLES)

        output_dir = OUTPUT_ROOT / "celeba" #input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, image_path in enumerate(sampled_images):
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                width, height = img.size

                if width < CROP_SIZE or height < CROP_SIZE:
                    scale = CROP_SIZE / min(width, height)
                    new_width = round(width * scale)
                    new_height = round(height * scale)
                    img = img.resize((new_width, new_height))
                    width, height = img.size

                left = (width - CROP_SIZE) // 2
                top = (height - CROP_SIZE) // 2
                right = left + CROP_SIZE
                bottom = top + CROP_SIZE

                img = img.crop((left, top, right, bottom))
                img.save(output_dir / f"{i:05d}.png")

        print(f"{input_dir.name}: saved {NUM_SAMPLES} center-cropped images")


if __name__ == '__main__':
    main()