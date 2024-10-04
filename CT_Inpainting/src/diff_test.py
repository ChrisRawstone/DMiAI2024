import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
from IPython.display import clear_output

clear_output()

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image
from huggingface_hub import model_info

# Ensure necessary imports are included
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image
from huggingface_hub import model_info
from tqdm import tqdm

# Ensure necessary imports are included
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("OFA-Sys/small-stable-diffusion-v0", subfolder="tokenizer")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
unet = UNet2DConditionModel.from_pretrained("OFA-Sys/small-stable-diffusion-v0", subfolder="unet").to(device) #torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained("OFA-Sys/small-stable-diffusion-v0", subfolder="vae").to(device) #torch_dtype=torch.float16).to(device)
text_encoder = CLIPTextModel.from_pretrained("OFA-Sys/small-stable-diffusion-v0", subfolder="text_encoder").to(device)

# Load the scheduler
scheduler = PNDMScheduler.from_pretrained("OFA-Sys/small-stable-diffusion-v0", subfolder="scheduler")

def save_and_visualize(epoch, unet, vae, text_encoder, tokenizer, scheduler, save = False):

    # Saving the trained models
    if save:
        unet.save_pretrained(f"path_to_save/unet_{epoch}")
        vae.save_pretrained(f"path_to_save/vae_{epoch}")
        text_encoder.save_pretrained(f"path_to_save/text_encoder_{epoch}")
        tokenizer.save_pretrained(f"path_to_save/tokenizer_{epoch}")
        scheduler.save_pretrained(f"path_to_save/scheduler_{epoch}")

    # Instantiate pipeline from finetuned elements
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        requires_safety_checker=False,  # Skip the safety checker
        safety_checker = None,
        feature_extractor = None
    ).to(device)
    

    # Specify promts
    prompts = [
        # 5 custom prompts
        "A ninja turtle riding a unicycle through a forest",
        "A giant frog sipping tea in a tranquil garden",
        "A masked warrior playing chess with a shadowy figure",
        "A fierce samurai cooking a barbecue under the moonlight",
        "A mysterious figure meditating on top of a floating mountain",
        "A ninja turtle riding a unicycle through a forest",
        "A giant frog sipping tea in a tranquil garden",
        "A masked warrior playing chess with a shadowy figure",
        "A fierce samurai cooking a barbecue under the moonlight",
        "A mysterious figure meditating on top of a floating mountain"
        ]

    # Create a grid layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns

    for i, prompt in enumerate(prompts):
        # Ensure that everything is in float16 to avoid dtype mismatch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pipe_out = pipeline(prompt, num_inference_steps=10)

        # Display the image
        img = pipe_out.images[0]

        ax = axes[i // 5, i % 5]  # Calculate grid position

        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Hide the axis

        # Add the corresponding prompt
        ax.set_title(prompts[i % len(prompts)], fontsize=12)

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.savefig(f'f_gen_epoch_{2}')
    
    # Freeze Text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

save_and_visualize(0, unet, vae, text_encoder, tokenizer, scheduler, save = False)