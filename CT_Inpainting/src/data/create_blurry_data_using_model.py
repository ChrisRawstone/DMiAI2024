from src.model_classes.model import UNet
import torch
from src.predict_model import predict, apply_only_to_mask
import os
from src.data.data_set_classes import BaseClass
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def create_data_set_based_on_predictions(path_to_model_output_folder="CT_Inpainting/models/2024-10-01/21-25-34"):
    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize or load the trained model
    model = UNet(in_channels=4, out_channels=1)  # Create an instance of the UNet model
    model.load_state_dict(torch.load(os.path.join(path_to_model_output_folder, "best_model.pth"), map_location=device))  # Load trained weights
    model.to(device)
    # go through the training data and make predictions
    training_dir = os.path.join(path_to_model_output_folder, "train_data")
    val_dir = os.path.join(path_to_model_output_folder, "val_data")
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts images to tensors with values in [0, 1]
                                    ])
    # read wheter to crop the mask or not from the config file used to train the model
    config_file = os.path.join(path_to_model_output_folder, ".hydra/config.yaml")
    config = OmegaConf.load(config_file)
    crop_mask = config.training_params.crop_mask

    for data_dir in [training_dir, val_dir]:
        data = BaseClass(data_dir=data_dir, transform=transform, crop_mask=crop_mask)
        # make folder for the generated data
        folder_for_generated_data = os.path.join(data_dir, "generated")
        os.makedirs(folder_for_generated_data, exist_ok=True)     
        for i in tqdm(range(len(data)), desc=f"Processing {data_dir}"):
            # identifer / filename of the image
            identifier = data.identifers[i]

            (inputs, labels) = data[i]
            # add batch dimension to the input, which is expected by the model
            # we add the batch dimension at the beginning, and its just 1 in this case
            inputs = inputs.unsqueeze(0).to(device)

            model.eval()
            outputs = model(inputs)
            output_tensor = torch.clamp(outputs, 0, 1)

            reconstructed_np = output_tensor[0, 0].detach().cpu().numpy() 
            reconstructed_image = reconstructed_np * 255.0

            # turn into PIL image
            reconstructed_image = Image.fromarray(reconstructed_image.astype('uint8'), 'L')
            # save the generated data
            save_path = os.path.join(folder_for_generated_data, f"generated_{identifier}.png")
            reconstructed_image.save(save_path)

if __name__ == "__main__":
    create_data_set_based_on_predictions()