import os
from PIL import Image
from sort_data_by_vertebrae import read_identifier_from_file
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def create_tensor_data_set(path_to_folder,save_data_set=False):
    """
    Create a tensor data set from a folder of images
    The foldershould contain the following subfolders:
    - Folder
        - corrupted
            - corrupted_001_5.png
        -ct
            - ct_001_5.png
        -mask
            - mask_001_5.png
        -tissue
            - tissue_001_5.png
        -vertebrae
            - vertebrae_001_5.txt                 
    # the dataset will have dim (n_samples, 3, 256, 256)

    """
    sub_directories_names = ['corrupted','mask', 'tissue', 'ct', 'vertebrae']
    sub_directories_paths = [os.path.join(path_to_folder, sub_dir) for sub_dir in sub_directories_names]

    #total number of samples
    n_samples = len(os.listdir(sub_directories_paths[0]))    
    # crate a tensor data set
    tensor_data_set = torch.zeros(n_samples, 3, 256, 256)
    tensor_ground_truth = torch.zeros(n_samples, 256, 256)

    # go through each file in the the vertebrae folder
    # vertebrae files
    vertebrae_file_names = os.listdir(sub_directories_paths[4])
    vertebrae_file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                                      
    for index,file_name in enumerate(vertebrae_file_names):
        assert file_name.endswith(".txt") # make sure it is a text file , mearning index 4 in the list
        # read from the file
        with open(os.path.join(sub_directories_paths[4], file_name), 'r') as f:
            # get the vertebrae number
            vertebrae_num = int(f.readline().strip())
        
        for i, folder in enumerate(sub_directories_paths):
            if not os.path.exists(folder):
                raise ValueError(f"Folder {folder} does not exist")       
            # first three folders are the one we need for training, 4th is the ground truth and 5th is the corrupted image
            if i == 4:
                continue
            else:
                img = Image.open(os.path.join(folder, read_identifier_from_file(file_name, sub_directories_names[i], ".png"))).convert('L')
                # visualize the image
                #plt.imshow(img, cmap='gray')
                #plt.savefig(f'CT_Inpainting/plots/{i}.png')
                img = transforms.ToTensor()(img)  # Directly convert the grayscale image to tensor        

                if sub_directories_names[i] == 'ct':
                    tensor_ground_truth[index, :, :] = img
                    # visualize the image
                    #plt.imshow(img, cmap='gray')
                    #plt.savefig(f'CT_Inpainting/plots/ground_truth.png')
                else:
                    tensor_data_set[index, i, :, :] = img 
                    #plt.imshow(tensor_data_set[vertebrae_num, i, :, :], cmap='gray')
                    #plt.savefig(f'CT_Inpainting/plots/{i}.png')           
            
    if save_data_set:
        torch.save(tensor_data_set, os.path.join(path_to_folder, 'tensor_data_set.pt'))
        torch.save(tensor_ground_truth, os.path.join(path_to_folder, 'tensor_label_set.pt'))

    return tensor_data_set

if __name__ == '__main__':
    for i in range(0, 25):
        tensor_data_set=create_tensor_data_set(f'CT_Inpainting/data_sorted_by_vertebrae/{i}',save_data_set=True)
    
        fig, axs = plt.subplots(1, 3)
        for j, ax in enumerate(axs):
            ax.imshow(tensor_data_set[0, j, :, :].numpy(), cmap='gray')
        plt.savefig(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/sample_plot.png')
        plt.close()





   

    
    




