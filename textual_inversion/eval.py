import os
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
import numpy as np
import yaml
import logging

with open("conf/evaluation.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
log_file = os.path.join(config["logs_dir"], "textual_inversion.log")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

logger.info("Reading config file")


def check_directory(dir_path: str) -> bool:
    """Check if directory exists and is not empty."""
    if not os.path.exists(dir_path):
        logger.error(f"Directory does not exist: {dir_path}")
        return False
    if not os.path.isdir(dir_path):
        logger.error(f"Path is not a directory: {dir_path}")
        return False
    if not any(Path(dir_path).rglob("*.png")):
        logger.error(f"Directory contains no PNG images: {dir_path}")
        return False
    return True


def load_dir_as_tensor(dir_path):
    if not check_directory(dir_path):
        raise ValueError(f"Invalid directory: {dir_path}")

    images = []
    dir_path = Path(dir_path)
    for image_path in dir_path.rglob("*.png"):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        image = np.moveaxis(image, -1, 0)

        image = np.clip(image, 0, 255).astype(np.uint8)
        images.append(image[None, ...])

    batch_tensor = torch.from_numpy(np.concatenate(images)).to(torch.uint8)
    return batch_tensor


def run_evaluation(target_image_dir: str, generated_image_dir: str):
    for dir_path in [target_image_dir, generated_image_dir]:
        if not check_directory(dir_path):
            raise ValueError(f"Invalid directory: {dir_path}")

    logger.info(f"Loading images from target directory: {target_image_dir}")
    target_images = load_dir_as_tensor(target_image_dir)
    logger.info(f"Loading images from generated directory: {generated_image_dir}")
    generated_images = load_dir_as_tensor(generated_image_dir)

    logger.info("Computing FID score...")
    fid = FrechetInceptionDistance()
    fid.update(target_images, real=True)
    fid.update(generated_images, real=False)
    fid_value = fid.compute().item()
    logger.info(f"FID score: {fid_value:.4f}")

    logger.info("Computing Inception Score...")
    inseption_score = InceptionScore()
    inseption_score.update(generated_images)
    result, _ = inseption_score.compute()
    ins_value = result.item()
    logger.info(f"Inception Score: {ins_value:.4f}")

    return fid_value, ins_value


def main():
    logger.info("Starting evaluation process...")
    target_images_dirs = config["data"]["target_images_dirs"]
    generated_images_dirs = config["data"]["generated_images_dirs"]

    if not check_directory(target_images_dirs):
        raise ValueError(f"Invalid target images directory: {target_images_dirs}")
    if not check_directory(generated_images_dirs):
        raise ValueError(f"Invalid generated images directory: {generated_images_dirs}")

    for target_image_dir, generated_images_dir in zip(
        os.listdir(target_images_dirs), os.listdir(generated_images_dirs)
    ):
        logger.info("Processing directory pair:")
        logger.info(f"Target: {target_image_dir}")
        logger.info(f"Generated: {generated_images_dir}")

        fid_value, ins_value = run_evaluation(
            os.path.join(target_images_dirs, target_image_dir),
            os.path.join(generated_images_dirs, generated_images_dir),
        )

    logger.info("Evaluation process completed.")


if __name__ == "__main__":
    main()
