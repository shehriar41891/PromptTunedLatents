import argparse
import logging
from utils import (
    stablediffusion_pipeline,
    get_text_embedding,
    condition_latent,
    generation
)
from config import Config
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main(prompt: str, output_path: str):
    logging.info("Initializing Stable Diffusion pipeline...")
    sd_pipeline = stablediffusion_pipeline()

    logging.info(f"Encoding prompt: \"{prompt}\"")
    text_embedding = get_text_embedding(prompt)

    logging.info("Conditioning latent vector using prompt embedding...")
    conditioned_latent_vector = condition_latent(text_embedding)

    logging.info("Generating image from conditioned latent space...")
    image = generation(conditioned_latent_vector, sd_pipeline)

    image.save(output_path)
    logging.info(f"Image saved to: {output_path}")

    image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-conditioned Stable Diffusion Generator")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A futuristic city skyline at sunset",
        help="Text prompt to guide image generation"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.png",
        help="File path to save the generated image"
    )

    args = parser.parse_args()
    main(prompt=args.prompt, output_path=args.output)
