import torch

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_INFERENCE_STEPS = 50
    CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
    CLIP_MODEL_PROCESSOR = "openai/clip-vit-base-patch16"
    STABLE_DIFFUSION_MODEL_ID = "CompVis/stable-diffusion-v1-4"
    OUTPUT_IMAGE_PATH = "outputs/generated_image.png"