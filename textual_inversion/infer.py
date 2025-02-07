import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import yaml
import shutil
import logging

with open("conf/inference.yaml", "r") as f:
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

# try:
#     scheduler = config["scheduler"]
# except KeyError:
#     raise NotImplementedError("Scheduler configuration is missing in the YAML file.")

# if scheduler == "DPM++ 2M" or scheduler == "DPM++ 2M Karras":
#     from diffusers import DPMSolverMultistepScheduler
# elif scheduler == "Euler Ancestral":
#     from diffusers import EulerAncestralDiscreteScheduler

HYPERPARAMETERS = {
    "num_inference_steps": config["hparams"]["num_inference_steps"],
    "guidance_scale": config["hparams"]["guidance_scale"],
    "num_generations": config["hparams"]["num_generations"],
}

BASIC_PROMPT = "a photo of a <placeholder>"


def run_inference(generated_images_dir: str):
    model_name = config["stable_diffusion"]["model_name"]

    generated_images_dir_name = generated_images_dir.split("/")[-1]

    if os.path.exists(generated_images_dir):
        shutil.rmtree(generated_images_dir)
    os.makedirs(generated_images_dir)

    embeddings_output_dir = os.path.join(
        config["embeddings"]["output_dir"], generated_images_dir_name
    )

    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # try float16
        ).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # if scheduler == "DPM++ 2M":
    #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # elif scheduler == "DPM++ 2M Karras":
    #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #         pipe.scheduler.config,
    #         use_karras_sigmas=True,
    #     )
    # elif scheduler == "Euler Ancestral":
    #     pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    #         pipe.scheduler.config
    #     )

    weight_name = "learned_embeds.bin"

    pipe.load_textual_inversion(
        embeddings_output_dir, weight_name=weight_name, local_files_only=True
    )

    pipe.safety_checker = None

    if generated_images_dir_name == "positive":
        placeholder_token = "<pcam_pos>"
    elif generated_images_dir_name == "negative":
        placeholder_token = "<pcam_neg>"

    logger.info(f"Embeddings output directory: {embeddings_output_dir}")
    logger.info(f"Generated images directory: {generated_images_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Placeholder token: {placeholder_token}")
    logger.info(f"Hyperparameters: {HYPERPARAMETERS}")

    logger.info(f"Saving images to {generated_images_dir}...")

    for i in range(HYPERPARAMETERS["num_generations"]):
        image = pipe(
            BASIC_PROMPT.replace("<placeholder>", placeholder_token),
            num_inference_steps=HYPERPARAMETERS["num_inference_steps"],
            guidance_scale=HYPERPARAMETERS["guidance_scale"],
        ).images[0]

        if generated_images_dir_name == "positive":
            file_name = "pcam_pos"
        elif generated_images_dir_name == "negative":
            file_name = "pcam_neg"

        logger.info(
            f"Saving image: {os.path.join(generated_images_dir, f'{file_name}_{i}.png')}"
        )
        image.save(os.path.join(generated_images_dir, f"{file_name}_{i}.png"))


def main():
    generated_images_dirs = config["data"]["generated_images_dirs"]

    logger.info("Running inference for the following images directories:")
    for generated_dir in os.listdir(generated_images_dirs):
        logger.info(generated_dir)

    for target_dir in os.listdir(generated_images_dirs):
        logger.info(f"Running inference for {target_dir}")
        run_inference(os.path.join(generated_images_dirs, target_dir))


if __name__ == "__main__":
    main()
