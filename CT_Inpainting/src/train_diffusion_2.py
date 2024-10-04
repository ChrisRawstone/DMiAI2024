import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from src.data.diffusion_dataset import CTInpaintingDiffusionDataset2
import matplotlib.pyplot as plt
from diffusers.utils import load_image, make_image_grid


### HELPER 
def save_and_visualize_old(dataloader, epoch, unet, vae, text_encoder, tokenizer, scheduler, save = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Saving the trained models
    if save:
        unet.save_pretrained(f"models/unet_{epoch}")
        vae.save_pretrained(f"models/vae_{epoch}")
        text_encoder.save_pretrained(f"models/text_encoder_{epoch}")
        tokenizer.save_pretrained(f"models/tokenizer_{epoch}")
        scheduler.save_pretrained(f"models/scheduler_{epoch}")

    # Instantiate pipeline from finetuned elements
    pipeline = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                requires_safety_checker=False,  # Skip the safety checker
                safety_checker = None,
                feature_extractor = None,
            ).to(device)
    


    # Get the first batch
    for batch in dataloader:
        corrupted, overlapping_corruption_mask, ct, _, _ = batch
        
        break
    
    
    # Define the rest of the inputs
    generator = torch.Generator("cuda").manual_seed(92)
    prompt = "Complete the image, by restoring the corrupted areas making it look like a real CT slice image of the human body"
    
    # Run through the pipeline to get the inpainted image
    image = pipeline(prompt=prompt, 
                     image=corrupted, 
                     mask_image=overlapping_corruption_mask, 
                     generator=generator, num_inference_steps = 100).images[0]
    
    
    # Change the corrupted and overlapping_corruption_mask tensors to PIL images
    corrupted = transforms.ToPILImage()(corrupted.squeeze(0).cpu())
    overlapping_corruption_mask = transforms.ToPILImage()(overlapping_corruption_mask.squeeze(0).cpu())
    ct_ground_truth = transforms.ToPILImage()(ct.squeeze(0).cpu())
    
    # Change the size back to 256x256
    corrupted = corrupted.resize((256, 256))
    overlapping_corruption_mask = overlapping_corruption_mask.resize((256, 256))
    image = image.resize((256, 256))
    ct_ground_truth = ct_ground_truth.resize((256, 256))
    
    
    grid_plot = make_image_grid([corrupted, overlapping_corruption_mask, image, ct_ground_truth], rows=1, cols=4)
    
    
    plt.imshow(grid_plot)
    plt.axis('off')  # Hide axes for better visualization
    plt.savefig(f"diffusion_generations/diffusion_gen_epoch_{epoch}.png", dpi = 300)
    
### HELPER 
def save_and_visualize(dataloader, epoch, unet, vae, text_encoder, tokenizer, scheduler, vertebrae_embedding, save=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Saving the trained models
    if save:
        unet.save_pretrained(f"models/unet_{epoch}")
        vae.save_pretrained(f"models/vae_{epoch}")
        text_encoder.save_pretrained(f"models/text_encoder_{epoch}")
        tokenizer.save_pretrained(f"models/tokenizer_{epoch}")
        scheduler.save_pretrained(f"models/scheduler_{epoch}")

    # Instantiate pipeline from finetuned elements
    pipeline = StableDiffusionInpaintPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    safety_checker=None,  # Set explicitly to None
                    feature_extractor=None,  # Set explicitly to None
                ).to(device)
    
    # Get the first batch
    for batch in dataloader:
        corrupted_ct, overlapping_corruption_mask, ct, tissue_image, vertebrae_number = batch
        corrupted_ct = corrupted_ct.to(device)
        overlapping_corruption_mask = overlapping_corruption_mask.to(device)
        tissue_image = tissue_image.to(device)
        vertebrae_number = vertebrae_number.to(device)
        break

    # Define the rest of the inputs
    generator = torch.Generator("cuda").manual_seed(92)
    prompt = "Complete the image, by restoring the corrupted areas making it look like a real CT slice image of the human body"
    
    # Get the text embeddings
    promt_embeds, negative_embeds = pipeline.encode_prompt(prompt = prompt, device = device, num_images_per_prompt = 1,do_classifier_free_guidance = True)
                                                        
    #input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids.repeat(corrupted_ct.shape[0], 1).to(device)
    #text_embeddings = text_encoder(input_ids)[0].to(dtype=torch.float32)

    # Get latent representation of the corrupted image via the VAE
    latents = vae.encode(corrupted_ct).latent_dist.sample() * vae.config.scaling_factor

    # Noise and mask latent
    mask_resized = torch.nn.functional.interpolate(overlapping_corruption_mask, size=(latents.shape[2], latents.shape[3]), mode="nearest")
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    masked_latents = noisy_latents * mask_resized + latents * (1 - mask_resized)

    # Embed the tissue map and vertebrae number
    tissue_image_3channel = tissue_image.repeat(1, 3, 1, 1)  # Convert 1-channel to 3-channel
    latent_tissue = vae.encode(tissue_image_3channel).latent_dist.sample() * vae.config.scaling_factor
    latent_tissue_flattened = latent_tissue.view(latent_tissue.shape[0], -1)

    # Project the flattened latent tissue to match the text embedding dimensions
    latent_tissue_projected = torch.nn.Linear(latent_tissue_flattened.shape[-1], promt_embeds.shape[-1]).to(device)
    latent_tissue_projected = latent_tissue_projected(latent_tissue_flattened).unsqueeze(1).repeat(1, promt_embeds.shape[1], 1)

    # Project vertebrae embedding to match text embedding dimensions
    vertebrae_embedded = vertebrae_embedding(vertebrae_number)
    vertebrae_projection = torch.nn.Linear(vertebrae_embedded.shape[-1], promt_embeds.shape[-1]).to(device)
    vertebrae_embedded_projected = vertebrae_projection(vertebrae_embedded).unsqueeze(1).repeat(1, promt_embeds.shape[1], 1)

 
    # Run inference using the pipeline
    image = pipeline(prompt=prompt,
                 image=corrupted_ct,
                 mask_image=overlapping_corruption_mask,
                 generator=generator,
                 num_inference_steps=100,
                 latent_tissue_projected = latent_tissue_projected,
                vertebrae_embedded_projected = vertebrae_embedded_projected,).images[0]
    
    # Convert tensors to PIL images for visualization
    corrupted_ct_img = transforms.ToPILImage()(corrupted_ct.squeeze(0).cpu())
    overlapping_corruption_mask_img = transforms.ToPILImage()(overlapping_corruption_mask.squeeze(0).cpu())
    ct_ground_truth_img = transforms.ToPILImage()(ct.squeeze(0).cpu())
    #image = transforms.ToPILImage()(image.squeeze(0).cpu())

    # Resize for visualization
    corrupted_ct_img = corrupted_ct_img.resize((256, 256))
    overlapping_corruption_mask_img = overlapping_corruption_mask_img.resize((256, 256))
    image = image.resize((256, 256))
    ct_ground_truth_img = ct_ground_truth_img.resize((256, 256))

    # Create grid and plot images
    grid_plot = make_image_grid([corrupted_ct_img, overlapping_corruption_mask_img, image, ct_ground_truth_img], rows=1, cols=4)

    # Plot the grid
    plt.imshow(grid_plot)
    plt.axis('off')  # Hide axes for better visualization
    plt.savefig(f"diffusion_generations/diffusion_gen_epoch_{epoch}.png", dpi=300)


from diffusers import StableDiffusionInpaintPipeline


class CustomStableDiffusionInpaintPipeline2(StableDiffusionInpaintPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, 
                 safety_checker=None, feature_extractor=None, image_encoder = None, requires_safety_checker=False):
        # Initialize parent class with required components
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, 
                 safety_checker=None, feature_extractor=None, image_encoder= None, requires_safety_checker=False)

        
            
    def __call__(self, prompt, image, mask_image, generator, num_inference_steps=50, encoder_hidden_states=None):
        # Step 1: Tokenize and encode the prompt (if needed)
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.device)

        # If no custom encoder_hidden_states are provided, fallback to using text embeddings
        if encoder_hidden_states is None:
            encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Step 2: Get latents from the VAE
        latents = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor

        # Step 3: Prepare the masked image latents
        mask_resized = F.interpolate(mask_image, size=(latents.shape[2], latents.shape[3]), mode="nearest")
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        masked_latents = noisy_latents * mask_resized + latents * (1 - mask_resized)

        # Step 4: Prepare UNet input
        unet_input = torch.cat([latents, masked_latents, mask_resized], dim=1)

        # Step 5: Pass through UNet with custom encoder_hidden_states
        noise_pred = self.unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # Step 6: Run denoising loop (this is simplified for illustration)
        for t in reversed(range(num_inference_steps)):
            noise_pred = self.unet(unet_input, torch.tensor([t]).to(self.device), encoder_hidden_states=encoder_hidden_states).sample

        # Step 7: Decode the latents to image
        decoded_image = self.vae.decode(latents).sample

        return decoded_image

from diffusers import StableDiffusionInpaintPipeline


 

############### SETUP DATA ################

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
dataset = CTInpaintingDiffusionDataset2(data_dir='data', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Take a tiny subset of the dataset for testing
subset_size = 100  # Define the size of the subset
subset_indices = torch.randperm(len(dataset))[:subset_size]
subset = torch.utils.data.Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=1, shuffle=True)

############# SETUP MODEL ################
path = "stabilityai/stable-diffusion-2-inpainting"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet").to(device) #torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained(path, subfolder="vae").to(device) #torch_dtype=torch.float16).to(device)
text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder").to(device)
scheduler = PNDMScheduler.from_pretrained(path, subfolder="scheduler")
vertebrae_embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=128).to(device)


# Freeze Text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)


# Instantiate pipeline from finetuned elements
pipeline_train = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,  # Set explicitly to None
            feature_extractor=None,  # Set explicitly to None
        ).to(device)


############# TRAINING PARAMETERS ################
optimizer = torch.optim.AdamW(
    list(unet.parameters()), 
    lr=1e-6
)
num_epochs = 100
accumulation_steps = 4
optimizer.zero_grad()

# Define promt to use
prompt = "Fill in the corrupted areas of the cross sectional CT slice of the human midsection with accurate anatomical details, including bones, organs, and soft tissue."


# ############ TRAINING LOOP ################
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    train_loss = 0.0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        # 1 Extract data from the batch and move to device
        corrupted_ct, mask, ct, tissue_image, vertebrae_number = batch
        corrupted_ct = corrupted_ct.to(device)
        mask = mask.to(device)
        ct = ct.to(device)
        tissue_image = tissue_image.to(device)
        vertebrae_number = vertebrae_number.to(device)
        
        # 2. Tokenixe and embed text prompt
        text_embeddings, negative_embeds = pipeline_train.encode_prompt(prompt = prompt, device = device, num_images_per_prompt = 1,do_classifier_free_guidance = True)

        #input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids.repeat(corrupted_ct.shape[0], 1).to(device)
        #text_embeddings = text_encoder(input_ids)[0].to(dtype=torch.float32)

        # 3. Pass corrupted CT through VAE to get latents
        latents = vae.encode(corrupted_ct).latent_dist.sample()* vae.config.scaling_factor

        # 4. Noise and mask latent
        mask_resized = torch.nn.functional.interpolate(mask, size=(latents.shape[2], latents.shape[3]),  mode="nearest")
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        masked_latents = noisy_latents * mask_resized + latents * (1 - mask_resized)  # Only noise in the masked regions

        # 5. Embed the tissue map and vertebrae number
        tissue_image_3channel = tissue_image.repeat(1, 3, 1, 1)  # Convert 1-channel to 3-channel
        latent_tissue = vae.encode(tissue_image_3channel).latent_dist.sample() * vae.config.scaling_factor

        # Flatten the latent_tissue to (batch_size, seq_len, latent_dim)
        latent_tissue_flattened = latent_tissue.view(latent_tissue.shape[0], -1)  # Shape: (batch_size, flattened_dim)

        # Now project the flattened latent tissue to match the text embedding dimensions
        latent_tissue_projected = torch.nn.Linear(latent_tissue_flattened.shape[-1], text_embeddings.shape[-1]).to(device)
        latent_tissue_projected = latent_tissue_projected(latent_tissue_flattened).unsqueeze(1).repeat(1, text_embeddings.shape[1], 1)

        # Project vertebrae_embedded to match the embedding dimension of text_embeddings
        vertebrae_embedded = vertebrae_embedding(vertebrae_number)
        vertebrae_projection = torch.nn.Linear(vertebrae_embedded.shape[-1], text_embeddings.shape[-1]).to(device)
        vertebrae_embedded_projected = vertebrae_projection(vertebrae_embedded).unsqueeze(1).repeat(1, text_embeddings.shape[1], 1)

        vertebrae_embedded_projected = vertebrae_embedding(vertebrae_number).unsqueeze(1).repeat(1, text_embeddings.shape[1], 1)  # Shape: (batch_size, seq_len, embedding_dim)
        vertebrae_embedded_projected = vertebrae_projection(vertebrae_embedded).unsqueeze(1).repeat(1, text_embeddings.shape[1], 1)

        # 6. Gather UNET Input
        unet_input = torch.cat([latents, masked_latents, mask_resized], dim=1)

        # 7. Define hidden states for cross-attention
        # Make sure all embeddings have the same shape (batch_size, seq_len, embedding_dim)
        #encoder_hidden_states = torch.cat([text_embeddings, latent_tissue_projected, vertebrae_embedded_projected], dim=0)
        encoder_hidden_states = text_embeddings + latent_tissue_projected + vertebrae_embedded_projected

        
        # 8. Predict noise using UNET

        noise_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample



        # 9. Determine target based on the prediction type
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

        # 10. Calculate loss 
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean") / accumulation_steps

        # AMP backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Gradient accumulation and optimization step
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()

    # Logging the loss
    avg_train_loss = train_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} completed with average loss: {avg_train_loss:.4f}")
    
    if epoch % 2 == 0:
      save_and_visualize(dataloader, epoch, unet, vae, text_encoder, tokenizer, scheduler, vertebrae_embedding, save = False)
