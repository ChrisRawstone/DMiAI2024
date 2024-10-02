# import shutil
# from pathlib import Path
# import pandas as pd

# # Define the original directories and CSV paths
# train_image_dir = Path("data/training")
# train_csv_path = Path("data/training.csv")
# val_image_dir = Path("data/validation16bit")
# val_csv_path = Path("data/validation.csv")

# # Define the combined directories and CSV path
# combined_image_dir = Path("data/combined_images")
# combined_image_dir.mkdir(parents=True, exist_ok=True)
# combined_csv_path = Path("data/combined.csv")

# def copy_and_rename_images(source_dir, prefix, dest_dir):
#     for img_path in source_dir.glob("*.tif"):
#         new_name = f"{prefix}_{img_path.name}"
#         dest_path = dest_dir / new_name
#         shutil.copy2(img_path, dest_path)

# def adjust_csv(csv_path, prefix):
#     df = pd.read_csv(csv_path)
#     # Ensure 'image_id' is zero-padded to three digits and add prefix
#     df['image_id'] = df['image_id'].astype(str).str.zfill(3)
#     df['image_id'] = df['image_id'].apply(lambda x: f"{prefix}_{x}")
#     return df

# # Copy and rename images
# copy_and_rename_images(train_image_dir, "train", combined_image_dir)
# copy_and_rename_images(val_image_dir, "val", combined_image_dir)

# # Adjust and combine CSV files
# train_df = adjust_csv(train_csv_path, "train")
# val_df = adjust_csv(val_csv_path, "val")
# combined_df = pd.concat([train_df, val_df], ignore_index=True)
# combined_df.to_csv(combined_csv_path, index=False)

# print(f"Combined images are saved in {combined_image_dir}")
# print(f"Combined CSV is saved as {combined_csv_path}")


import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define directories and files
data_dir = 'data/combined'
csv_file = 'data/combined.csv'
train_dir = 'data/training_val'
test_dir = 'data/test_val'
train_csv = 'data/training_val.csv'
test_csv = 'data/test_val.csv'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read and shuffle the CSV file
df = pd.read_csv(csv_file)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataframe

# Extract image IDs and labels
image_ids = df['image_id']
labels = df['is_homogenous']

# Stratified split
train_ids, test_ids, y_train, y_test = train_test_split(
    image_ids, labels, test_size=0.2, random_state=42, stratify=labels
)

# Function to copy and rename images
def process_images(ids, labels, dest_dir, prefix):
    data = []
    for idx, (img_id, label) in enumerate(zip(ids, labels), 1):
        src = os.path.join(data_dir, f"{img_id}.tif")
        dst_filename = f"{idx:03d}.tif"
        dst = os.path.join(dest_dir, dst_filename)
        shutil.copy(src, dst)
        data.append({'image_id': dst_filename[:-4], 'is_homogenous': label})
    return data

# Process training images
train_data = process_images(train_ids, y_train, train_dir, 'train')

# Process test images
test_data = process_images(test_ids, y_test, test_dir, 'val')

# Save CSV files
pd.DataFrame(train_data).to_csv(train_csv, index=False)
pd.DataFrame(test_data).to_csv(test_csv, index=False)