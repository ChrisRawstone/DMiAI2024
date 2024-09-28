import os

#def read_and_save_data_file(path_to_org_data_file):

def read_identifier_from_file(vertebrae_file_path,prefix,new_file_extension):
    new_file_name = vertebrae_file_path.replace('.txt', new_file_extension)
    new_file_name = new_file_name.replace("vertebrae", prefix)
    return new_file_name

    


if __name__ == '__main__':
    # create a new directory
    os.makedirs('CT_Inpainting/data_sorted_by_vertebrae', exist_ok=True)
    # make 25 directories for each vertebrae
    for i in range(0, 25):
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}', exist_ok=True)
        #within that folder make 5 folders, corrupted,ct,mask,tissue,vertebrae
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/corrupted', exist_ok=True)
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/ct', exist_ok=True)
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/mask', exist_ok=True)
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/tissue', exist_ok=True)
        os.makedirs(f'CT_Inpainting/data_sorted_by_vertebrae/{i}/vertebrae', exist_ok=True)

    # copy the orginal data to correct new folder. 
    for file in os.listdir('CT_Inpainting/data/vertebrae'):
        # read from the file
        with open(f'CT_Inpainting/data/vertebrae/{file}', 'r') as f:
            # get the vertebrae number
            vertebrae_num = int(f.readline().strip())
            os.system(f'cp CT_Inpainting/data/vertebrae/{file} CT_Inpainting/data_sorted_by_vertebrae/{vertebrae_num}/vertebrae')
            os.system(f'cp CT_Inpainting/data/corrupted/{read_identifier_from_file(file,"corrupted",".png")} CT_Inpainting/data_sorted_by_vertebrae/{vertebrae_num}/corrupted')
            os.system(f'cp CT_Inpainting/data/ct/{read_identifier_from_file(file,"ct",".png")} CT_Inpainting/data_sorted_by_vertebrae/{vertebrae_num}/ct')
            os.system(f'cp CT_Inpainting/data/mask/{read_identifier_from_file(file,"mask",".png")} CT_Inpainting/data_sorted_by_vertebrae/{vertebrae_num}/mask')
            os.system(f'cp CT_Inpainting/data/tissue/{read_identifier_from_file(file,"tissue",".png")} CT_Inpainting/data_sorted_by_vertebrae/{vertebrae_num}/tissue')
            
            
