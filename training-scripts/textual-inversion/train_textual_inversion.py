import os
import sys
import argparse
import yaml

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

logger.info("Reading config file")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HYPERPARAMETERS = {
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_train_steps": 2000,
    "learning_rate": 5.0e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42,
}

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def run_training(
    target_images_dir,
    model_output_dir,
    model_name,
    placeholder_token,
    initializer_token,
    hyperparameters,
    save_as_full_pipeline,
    no_safe_serialization,
    resume_checkpoint=None,
):
    """Run the textual inversion training."""
    command = f'accelerate launch /homes/obaldoni/bio_project/training-scripts/textual-inversion/textual_inversion.py \
        --pretrained_model_name_or_path={model_name} \
        --train_data_dir={target_images_dir} \
        --learnable_property="object" \
        --placeholder_token="{placeholder_token}" --initializer_token="{initializer_token}" \
        --resolution={hyperparameters["resolution"]} \
        --train_batch_size={hyperparameters["train_batch_size"]} \
        --gradient_accumulation_steps={hyperparameters["gradient_accumulation_steps"]} \
        --max_train_steps={hyperparameters["max_train_steps"]} \
        --learning_rate={hyperparameters["learning_rate"]} --scale_lr \
        --lr_scheduler="{hyperparameters["lr_scheduler"]}" \
        --lr_warmup_steps={hyperparameters["lr_warmup_steps"]} \
        --enable_xformers_memory_efficient_attention \
        --checkpoints_total_limit=1 \
        --seed={hyperparameters["seed"]} \
        --output_dir={model_output_dir}'

    if save_as_full_pipeline:
        command += " --save_as_full_pipeline"
    if no_safe_serialization:
        command += " --no_safe_serialization"
    if resume_checkpoint:
        command += f" --resume_from_checkpoint={resume_checkpoint}"

    os.system(command)

    # Return whether the training was successful
    weight_name = (
        "learned_embeds.bin" if no_safe_serialization else "learned_embeds.safetensors"
    )
    return os.path.exists(f"{model_output_dir}/{weight_name}")


def main(
    target_images_dir,
    initializer_token=None,
    model_output_dir=None,
    model_name=DEFAULT_MODEL_NAME,
    placeholder_token="<*>",
    generated_images_dir=None,
    no_train=False,
    train_log="training.log",
    save_as_full_pipeline=False,
    no_safe_serialization=False,
    resume_checkpoint=None,
):
    target_images_dir_name = target_images_dir.split("/")[-1]

    if model_output_dir is None:
        model_output_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/textual_inversion_embeds/{target_images_dir_name}"

    if generated_images_dir is None:
        generated_images_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/generated_data/{target_images_dir_name}"

    # Clear and recreate the generated images directory
    os.system(f"rm -rf {generated_images_dir}")
    os.system(f"mkdir -p {generated_images_dir}")

    # If initializer token is not provided, use the directory name
    if initializer_token is None:
        initializer_token = target_images_dir_name

    print(f"Model output directory: {model_output_dir}\n")
    print(f"Generated images directory: {generated_images_dir}\n")
    print(f"Model name: {model_name}\n")
    print(f"Placeholder token: {placeholder_token}\n")
    print(f"Initializer token: {initializer_token}\n")
    print(f"Hyperparameters: {HYPERPARAMETERS}\n")

    with open(train_log, "w") as f:
        f.write(f"Model output directory: {model_output_dir}\n")
        f.write(f"Generated images directory: {generated_images_dir}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Placeholder token: {placeholder_token}\n")
        f.write(f"Initializer token: {initializer_token}\n")
        f.write(f"Hyperparameters: {HYPERPARAMETERS}\n")

    if no_train:
        print(f"Skipping training for the target images in {target_images_dir}.\n")
        if not os.path.exists(model_output_dir):
            sys.exit(
                f"Model output directory {model_output_dir} does not exist. Exiting."
            )

        with open(train_log, "a") as f:
            f.write(
                f"Skipping training for the target images in {target_images_dir}.\n"
            )

        model_training_successful = True
    elif not resume_checkpoint:
        print(
            f"Running textual inversion training for the target images in {target_images_dir}.\n"
        )
        with open(train_log, "a") as f:
            f.write(
                f"Running textual inversion training for the target images in {target_images_dir}.\n"
            )

        model_training_successful = run_training(
            target_images_dir,
            model_output_dir,
            model_name,
            placeholder_token,
            initializer_token,
            HYPERPARAMETERS,
            save_as_full_pipeline,
            no_safe_serialization,
        )
    else:
        print(f"Resuming training from checkpoint {resume_checkpoint}.\n")
        with open(train_log, "a") as f:
            f.write(f"Resuming training from checkpoint {resume_checkpoint}.\n")

        model_training_successful = run_training(
            target_images_dir,
            model_output_dir,
            model_name,
            placeholder_token,
            initializer_token,
            HYPERPARAMETERS,
            save_as_full_pipeline,
            no_safe_serialization,
            resume_checkpoint,
        )

    if not model_training_successful:
        with open(train_log, "a") as f:
            f.write("Model training failed. Exiting.\n")
        sys.exit("Model training failed. Exiting.")

    with open(train_log, "a") as f:
        f.write("Training completed successfully\n")


def run_textual_inversion(target_images_dir: str):
    target_images_dir_name = target_images_dir.split("/")[-1]

    model_output_dir = config.get("model_output_dir", None)
    if model_output_dir is None:
        model_output_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/textual_inversion_embeds/{target_images_dir_name}"

    generated_images_dir = config.get("generated_images_dir", None)
    if generated_images_dir is None:
        generated_images_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/generated_data/{target_images_dir_name}"

    # Clear and recreate the generated images directory
    os.system(f"rm -rf {generated_images_dir}")
    os.system(f"mkdir -p {generated_images_dir}")

    initializer_token = config.get("initializer_token", None)
    if initializer_token is None:
        initializer_token = target_images_dir_name

    log_file = f"{config['train_log_dir']}/{target_dir}.log"

    logger.info(f"Model output directory: {model_output_dir}\n")
    logger.info(f"Generated images directory: {generated_images_dir}\n")
    logger.info(f"Model name: {model_name}\n")
    logger.info(f"Placeholder token: {placeholder_token}\n")
    logger.info(f"Initializer token: {initializer_token}\n")
    logger.info(f"Hyperparameters: {HYPERPARAMETERS}\n")


if __name__ == "__main__":
    target_images_dirs = config["target_images_dirs"]

    # target_images_dirs = r"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/data"
    logger.info("Running training for the following target images directories:")
    for target_dir in os.listdir(target_images_dirs):
        logger.info(target_dir)

    for target_dir in os.listdir(target_images_dirs):
        logger.info(f"Running training for {target_dir}")
        run_textual_inversion(os.path.join(target_images_dirs, target_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_images_dir", type=str, required=True)
    parser.add_argument("--logfile", type=str, required=False, default="training.log")
    parser.add_argument("--model_output_dir", type=str, required=False)
    parser.add_argument(
        "--model_name", type=str, required=False, default=DEFAULT_MODEL_NAME
    )
    parser.add_argument("--placeholder_token", type=str, required=False, default="<*>")
    parser.add_argument("--initializer_token", type=str, required=False)
    parser.add_argument("--generated_images_dir", type=str, required=False)
    parser.add_argument("--no_train", type=bool, required=False, default=False)
    parser.add_argument(
        "--save_as_full_pipeline", type=bool, required=False, default=False
    )
    parser.add_argument(
        "--no_safe_serialization", type=bool, required=False, default=False
    )
    parser.add_argument("--resume_checkpoint", type=str, required=False)
    args = parser.parse_args()

    main(
        args.target_images_dir,
        args.initializer_token,
        args.model_output_dir,
        args.model_name,
        args.placeholder_token,
        args.generated_images_dir,
        args.no_train,
        args.logfile,
        args.save_as_full_pipeline,
        args.no_safe_serialization,
        args.resume_checkpoint,
    )
