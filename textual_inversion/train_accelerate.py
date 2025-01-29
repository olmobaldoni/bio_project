import os
import shutil
from typing import Any
import yaml

import logging

with open("conf/textual_inversion.yaml", "r") as f:
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

HYPERPARAMETERS = {
    "resolution": config["hparams"]["resolution"],
    "train_batch_size": config["hparams"]["train_batch_size"],
    "gradient_accumulation_steps": config["hparams"]["gradient_accomulation_steps"],
    "max_train_steps": config["hparams"]["max_train_steps"],
    "learning_rate": config["hparams"]["learning_rate"],
    "lr_scheduler": config["hparams"]["lr_scheduler"],
    "lr_warmup_steps": config["hparams"]["lr_warmup_steps"],
    "num_inference_steps": config["hparams"]["num_inference_steps"],
    "guidance_scale": config["hparams"]["guidance_scale"],
    "seed": config["hparams"]["seed"],
}


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

    if placeholder_token == "<pcam_pos>":
        weight_name = (
            "pcam_pos.bin" if no_safe_serialization else "pcam_pos.safetensors"
        )
    elif placeholder_token == "<pcam_neg>":
        weight_name = (
            "pcam_neg.bin" if no_safe_serialization else "pcam_neg.safetensors"
        )        

    return os.path.exists(f"{model_output_dir}/{weight_name}")


# def main(
#     target_images_dir,
#     initializer_token=None,
#     model_output_dir=None,
#     model_name=DEFAULT_MODEL_NAME,
#     placeholder_token="<*>",
#     generated_images_dir=None,
#     no_train=False,
#     train_log="training.log",
#     save_as_full_pipeline=False,
#     no_safe_serialization=False,
#     resume_checkpoint=None,
# ):
#     target_images_dir_name = target_images_dir.split("/")[-1]

#     if model_output_dir is None:
#         model_output_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/textual_inversion_embeds/{target_images_dir_name}"

#     if generated_images_dir is None:
#         generated_images_dir = f"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/generated_data/{target_images_dir_name}"

#     # Clear and recreate the generated images directory
#     os.system(f"rm -rf {generated_images_dir}")
#     os.system(f"mkdir -p {generated_images_dir}")

#     # If initializer token is not provided, use the directory name
#     if initializer_token is None:
#         initializer_token = target_images_dir_name

#     print(f"Model output directory: {model_output_dir}\n")
#     print(f"Generated images directory: {generated_images_dir}\n")
#     print(f"Model name: {model_name}\n")
#     print(f"Placeholder token: {placeholder_token}\n")
#     print(f"Initializer token: {initializer_token}\n")
#     print(f"Hyperparameters: {HYPERPARAMETERS}\n")

#     with open(train_log, "w") as f:
#         f.write(f"Model output directory: {model_output_dir}\n")
#         f.write(f"Generated images directory: {generated_images_dir}\n")
#         f.write(f"Model name: {model_name}\n")
#         f.write(f"Placeholder token: {placeholder_token}\n")
#         f.write(f"Initializer token: {initializer_token}\n")
#         f.write(f"Hyperparameters: {HYPERPARAMETERS}\n")

#     if no_train:
#         print(f"Skipping training for the target images in {target_images_dir}.\n")
#         if not os.path.exists(model_output_dir):
#             sys.exit(
#                 f"Model output directory {model_output_dir} does not exist. Exiting."
#             )

#         with open(train_log, "a") as f:
#             f.write(
#                 f"Skipping training for the target images in {target_images_dir}.\n"
#             )

#         model_training_successful = True
#     elif not resume_checkpoint:
#         print(
#             f"Running textual inversion training for the target images in {target_images_dir}.\n"
#         )
#         with open(train_log, "a") as f:
#             f.write(
#                 f"Running textual inversion training for the target images in {target_images_dir}.\n"
#             )

#         model_training_successful = run_training(
#             target_images_dir,
#             model_output_dir,
#             model_name,
#             placeholder_token,
#             initializer_token,
#             HYPERPARAMETERS,
#             save_as_full_pipeline,
#             no_safe_serialization,
#         )
#     else:
#         print(f"Resuming training from checkpoint {resume_checkpoint}.\n")
#         with open(train_log, "a") as f:
#             f.write(f"Resuming training from checkpoint {resume_checkpoint}.\n")

#         model_training_successful = run_training(
#             target_images_dir,
#             model_output_dir,
#             model_name,
#             placeholder_token,
#             initializer_token,
#             HYPERPARAMETERS,
#             save_as_full_pipeline,
#             no_safe_serialization,
#             resume_checkpoint,
#         )

#     if not model_training_successful:
#         with open(train_log, "a") as f:
#             f.write("Model training failed. Exiting.\n")
#         sys.exit("Model training failed. Exiting.")

#     with open(train_log, "a") as f:
#         f.write("Training completed successfully\n")


def run_accelerate(
    target_images_dir: str,
    embeddings_output_dir: str,
    model_name: str,
    initializer_token: str,
    placeholder_token: str,
    hyperparameters: dict[str, Any],
    save_as_full_pipeline: bool = False,
    no_safe_serialization: bool = False,
):
    command = f'accelerate launch /homes/obaldoni/bio_project/textual_inversion/src/textual_inversion.py \
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
        --output_dir={embeddings_output_dir}'

    if save_as_full_pipeline:
        command += " --save_as_full_pipeline"
    if no_safe_serialization:
        command += " --no_safe_serialization"

    os.system(command)

    if placeholder_token == "<pcam_pos>":
        weight_name = (
            "pcam_pos.bin" if no_safe_serialization else "pcam_pos.safetensors"
        )
    elif placeholder_token == "<pcam_neg>":
        weight_name = (
            "pcam_neg.bin" if no_safe_serialization else "pcam_neg.safetensors"
        )        

    return os.path.exists(f"{embeddings_output_dir}/{weight_name}")


def run_textual_inversion(target_images_dir: str):
    model_name = config["stable_diffusion"]["model_name"]

    target_images_dir_name = target_images_dir.split("/")[-1]

    embeddings_output_dir = os.path.join(
        config["embeddings"]["output_dir"], target_images_dir_name
    )
    if os.path.exists(embeddings_output_dir):
        shutil.rmtree(embeddings_output_dir)
    os.makedirs(embeddings_output_dir)

    generated_images_dir = os.path.join(
        config["data"]["generated_images_dir"], target_images_dir_name
    )
    if os.path.exists(generated_images_dir):
        shutil.rmtree(generated_images_dir)
    os.makedirs(generated_images_dir)

    initializer_token = target_images_dir_name
    if initializer_token == "positive":
        placeholder_token = "<pcam_pos>"
    elif initializer_token == "negative":
        placeholder_token = "<pcam_neg>"

    logger.info(f"Target images directory path: {target_images_dir}")
    logger.info(f"Target images directory name: {target_images_dir_name}")
    logger.info(f"Embeddings output directory: {embeddings_output_dir}")
    logger.info(f"Generated images directory: {generated_images_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Initializer token: {initializer_token}")
    logger.info(f"Placeholder token: {placeholder_token}")
    logger.info(f"Hyperparameters: {HYPERPARAMETERS}")

    train = run_accelerate(
        target_images_dir=target_images_dir,
        embeddings_output_dir=embeddings_output_dir,
        model_name=model_name,
        initializer_token=initializer_token,
        placeholder_token=placeholder_token,
        hyperparameters=HYPERPARAMETERS,
        save_as_full_pipeline=False,
        no_safe_serialization=False
    )

    if train:
        logger.info("Training completed successfully")
    else:
        logger.info("Training failed")

def main():
    target_images_dirs = config["data"]["target_images_dirs"]

    logger.info("Running training for the following target images directories:")
    for target_dir in os.listdir(target_images_dirs):
        logger.info(target_dir)

    for target_dir in os.listdir(target_images_dirs):
        logger.info(f"Running training for {target_dir}")
        run_textual_inversion(os.path.join(target_images_dirs, target_dir))


if __name__ == "__main__":
    main()
