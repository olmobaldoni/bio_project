import datasets
import os
import yaml
import shutil
from PIL import Image
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

with open("conf/data_preparation.yaml", "r") as f:
    config = yaml.safe_load(f)


def main():
    try:
        positive_dir = config["data"]["patchcamelyon"]["positive_dir"]
        negative_dir = config["data"]["patchcamelyon"]["negative_dir"]
    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        return

    # Clear and recreate the positive images directory
    if os.path.exists(positive_dir):
        shutil.rmtree(positive_dir)
    os.makedirs(positive_dir)

    # Clear and recreate the negative images directory
    if os.path.exists(negative_dir):
        shutil.rmtree(negative_dir)
    os.makedirs(negative_dir)

    # Load PatchCamelyon dataset from HF Hub
    ds = datasets.load_dataset(
        "1aurent/PatchCamelyon", revision="main", split="test", streaming=True
    )

    pos_count = 0
    neg_count = 0

    new_size = (512, 512)

    for item in ds:
        if pos_count < 100 and item["label"]:
            image = item["image"]
            image = image.resize(new_size, resample=Image.BICUBIC)
            image.save(os.path.join(positive_dir, f"pcam_pos_{pos_count}.png"))
            pos_count += 1
        elif neg_count < 100 and not item["label"]:
            image = item["image"]
            image = image.resize(new_size, resample=Image.BICUBIC)
            image.save(os.path.join(negative_dir, f"pcam_neg_{neg_count}.png"))
            neg_count += 1

        if pos_count == 100 and neg_count == 100:
            logger.info("100 positive and 100 negative images generated")
            break


if __name__ == "__main__":
    main()
