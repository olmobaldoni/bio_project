import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
import numpy as np
import yaml
import shutil
import logging
from itertools import product

# Load configurations
with open("conf/inference.yaml", "r") as f:
    inference_config = yaml.safe_load(f)

with open("conf/evaluation.yaml", "r") as f:
    eval_config = yaml.safe_load(f)

# Logging setup
log_file = os.path.join(inference_config["logs_dir"], "textual_inversion.log")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)
logger.info("Reading config files")

BASIC_PROMPT = "a photo of a <placeholder>"

def get_parameter_combinations():
    num_inference_steps = inference_config["hparams"]["num_inference_steps"]
    guidance_scale = inference_config["hparams"]["guidance_scale"]
    num_generations = inference_config["hparams"]["num_generations"]
    
    for steps, scale in product(num_inference_steps, guidance_scale):
        yield {
            "num_inference_steps": steps,
            "guidance_scale": scale,
            "num_generations": num_generations
        }

def check_directory(dir_path: str) -> bool:
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

def evaluate_generations(target_images_dirs: str, generated_images_dirs: str):
    logger.info("Evaluating generations")
    results = {}
    
    for target_dir, gen_dir in zip(
        os.listdir(target_images_dirs), os.listdir(generated_images_dirs)
    ):
        target_path = os.path.join(target_images_dirs, target_dir)
        generated_path = os.path.join(generated_images_dirs, gen_dir)
        
        logger.info(f"Processing:\nTarget: {target_path}\nGenerated: {generated_path}")
        
        target_images = load_dir_as_tensor(target_path)
        generated_images = load_dir_as_tensor(generated_path)

        # Calculate FID
        fid = FrechetInceptionDistance()
        fid.update(target_images, real=True)
        fid.update(generated_images, real=False)
        fid_value = fid.compute().item()

        # Calculate Inception Score
        inception_score = InceptionScore()
        inception_score.update(generated_images)
        ins_value, _ = inception_score.compute()
        
        results[gen_dir] = {
            "fid": fid_value,
            "inception_score": ins_value.item()
        }
        
        logger.info(f"Results for {gen_dir}:")
        logger.info(f"FID Score: {fid_value:.4f}")
        logger.info(f"Inception Score: {ins_value:.4f}")
    
    return results

def run_inference(pipe, generated_images_dir: str, params: dict):
    generated_images_dir_name = generated_images_dir.split("/")[-1]
    
    # Reset directory
    if os.path.exists(generated_images_dir):
        shutil.rmtree(generated_images_dir)
    os.makedirs(generated_images_dir)

    if generated_images_dir_name == "positive":
        placeholder_token = "<pcam_pos>"
    elif generated_images_dir_name == "negative":
        placeholder_token = "<pcam_neg>"

    logger.info(f"Generating images with parameters: {params}")
    logger.info(f"Saving to: {generated_images_dir}")

    for i in range(params["num_generations"]):
        image = pipe(
            BASIC_PROMPT.replace("<placeholder>", placeholder_token),
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
        ).images[0]

        file_name = f"pcam_{generated_images_dir_name}_{i}.png"
        output_path = os.path.join(generated_images_dir, file_name)
        logger.info(f"Saving image: {output_path}")
        image.save(output_path)

def main():
    model_name = inference_config["stable_diffusion"]["model_name"]
    generated_images_dirs = inference_config["data"]["generated_images_dirs"]
    target_images_dirs = eval_config["data"]["target_images_dirs"]
    
    # Initialize the model only once
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    
    # For each parameter combination
    for params in get_parameter_combinations():
        logger.info(f"\nStarting generation with parameters: {params}")
        
        # Generate images for each directory (positive/negative)
        for target_dir in os.listdir(generated_images_dirs):
            gen_dir_path = os.path.join(generated_images_dirs, target_dir)
            embeddings_path = os.path.join(
                inference_config["embeddings"]["output_dir"], 
                target_dir,
                "learned_embeds.bin"
            )
            
            # Load the specific embedding
            pipe.load_textual_inversion(
                embeddings_path, 
                weight_name="learned_embeds.bin", 
                local_files_only=True
            )
            
            # Generate the images
            run_inference(pipe, gen_dir_path, params)
        
        # Evaluate generations for this parameter combination
        results = evaluate_generations(target_images_dirs, generated_images_dirs)
        
        # Save results
        results_dir = os.path.join(inference_config["logs_dir"], "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir, 
            f"results_steps_{params['num_inference_steps']}_scale_{params['guidance_scale']}.yaml"
        )
        
        with open(results_file, 'w') as f:
            yaml.dump({
                "parameters": params,
                "results": results
            }, f)
        
        logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()