from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
from config import Config

clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL_ID)
clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_PROCESSOR)


def stablediffusion_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(Config.STABLE_DIFFUSION_MODEL_ID)
    return pipe.to(Config.DEVICE)


def get_text_embedding(prompt):
    inputs = clip_processor(text=prompt, return_tensors="pt", padding=True)
    text_embeddings = clip_model.get_text_features(**inputs)
    return text_embeddings

def condition_latent(text_embedding):
    latent_vector = torch.randn((1, 4, 64, 64)).to(Config.DEVICE) 
    text_embedding_scaled = F.interpolate(text_embedding, size=(64, 64), mode="bilinear")  
    conditioned_latent_vector = latent_vector + text_embedding_scaled.to("cuda")
    
    return conditioned_latent_vector

def generation(conditioned_latent_vector,pipe):
    num_inference_steps = Config.NUM_INFERENCE_STEPS
    image = pipe.decode_latents(conditioned_latent_vector, num_inference_steps=num_inference_steps).images[0]
    
    return image