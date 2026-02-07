from tqdm import tqdm
import os

from torchvision.datasets import CIFAR10

def save_cifar_as_jpeg(root: str = "data", split: str = "train", out_dir: str = "cifar10", quality: int = 95):
    train = (split.lower() == "train")
    ds = CIFAR10(root=root, train=train, download=True)

    out_dir = os.path.join(out_dir, "train" if train else "test")
    os.makedirs(out_dir, exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(ds, desc=f"Saving {split} images")):
        # 例: 000123_label7.jpg
        fname = os.path.join(out_dir, f"{idx:06d}_label{label}.jpg")
        img.save(
            fname,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True
        )

if __name__ == "__main__":
    save_cifar_as_jpeg(split="train", quality=95)
    save_cifar_as_jpeg(split="test", quality=95)
