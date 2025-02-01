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


def create_splits(path_dir: str, ds):
    dir_name = path_dir.split("/")[-1]
    logger.info(f"Starting to process images for {dir_name}")
    new_size = (512, 512)

    if dir_name == "cardiomegaly":
        logger.info("Filtering for Cardiomegaly positive frontal images")
        filtered_ds = ds.filter(
            lambda x: x["Frontal/Lateral"] == 0 and x["Cardiomegaly"] == 1
        )
    elif dir_name == "edema":
        logger.info("Filtering for Edema positive frontal images")
        filtered_ds = ds.filter(lambda x: x["Frontal/Lateral"] == 0 and x["Edema"] == 3)
    elif dir_name == "pneumonia":
        logger.info("Filtering for Pneumonia positive frontal images")
        filtered_ds = ds.filter(
            lambda x: x["Frontal/Lateral"] == 0 and x["Pneumonia"] == 3
        )
    elif dir_name == "fracture":
        logger.info("Filtering for Fracture positive frontal images")
        filtered_ds = ds.filter(
            lambda x: x["Frontal/Lateral"] == 0 and x["Fracture"] == 3
        )

    ds_iter = iter(filtered_ds)
    processed_images = 0

    logger.info(f"Starting to save images to {path_dir}")
    for i in range(100):
        try:
            item = next(ds_iter)
            image = item["image"]
            image = image.resize(new_size, resample=Image.BICUBIC)
            image.save(os.path.join(path_dir, f"chexpert_{dir_name}_{i}.png"))
            processed_images += 1
            if processed_images % 10 == 0:
                logger.info(f"Processed {processed_images} images for {dir_name}")
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")

    logger.info(f"Finished processing {processed_images} images for {dir_name}")
    return processed_images


def main():
    logger.info("Starting CheXpert data preparation")
    try:
        dirs = config["data"]["chexpert"]
        logger.info(f"Found {len(dirs)} directories in configuration")
    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        return

    logger.info("Loading CheXpert dataset")
    ds = datasets.load_dataset(
        "danjacobellis/chexpert",
        revision="main",
        split="train",
        streaming=True,
    )

    for path_dir in dirs.values():
        if os.path.exists(path_dir):
            logger.info(f"Removing existing directory: {path_dir}")
            shutil.rmtree(path_dir)
        logger.info(f"Creating directory: {path_dir}")
        os.makedirs(path_dir)

    for path_dir in dirs.values():
        create_splits(path_dir=path_dir, ds=ds)

    logger.info("CheXpert data preparation completed")


if __name__ == "__main__":
    main()
