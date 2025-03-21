import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURATION ---
image_folder = "CNN_training_png/"  # Folder containing PNG images
output_folder = "CNN_training_png_expanded/"  # Folder to save flipped images
npz_file = "processed_data.npz"  # Original dataset, created by the Extract_curves_and_E_c
output_npz_file = "processed_data_expanded.npz"  # New dataset

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# --- IMAGE PROCESSING FUNCTION ---
def flip_and_save(image_path, output_folder):
    """Flips an image along x, y, and z axes and saves them with new names."""
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Get filename without extension
    
    # Copy the original image to the new folder
    shutil.copy(image_path, os.path.join(output_folder, f"{base_name}.png"))

    flipped_images = {
        "_x": img.transpose(Image.FLIP_LEFT_RIGHT),  # Flip along x-axis
        "_y": img.transpose(Image.FLIP_TOP_BOTTOM),  # Flip along y-axis
        "_z": img.rotate(180)  # Rotate 180 degrees (z-axis flip)
    }
    
    new_names = []
    for suffix, flipped_img in flipped_images.items():
        new_name = f"{base_name}{suffix}.png"
        flipped_img.save(os.path.join(output_folder, new_name))
        new_names.append(base_name + suffix)  # Store new names for dataset expansion
    
    return base_name, new_names

# --- LOAD EXISTING DATASET ---
data = np.load(npz_file, allow_pickle=True)

curves_batch2 = data["curves_batch2"].item()
curves_batch3 = data["curves_batch3"].item()
# curves_batch4 = data["curves_batch4"].item()
e_com_batch2 = data["e_com_batch2"].item()
e_com_batch3 = data["e_com_batch3"].item()
# e_com_batch4 = data["e_com_batch4"].item()

# --- DATA AUGMENTATION FUNCTION ---
def augment_data(original_dict, original_names, new_names):
    """Duplicates data for only the corresponding batch."""
    augmented_dict = {}
    for name, value in original_dict.items():
        augmented_dict[name] = value  # Keep original
        if name in original_names:  # Only add flipped versions for existing names
            for suffix in ["_x", "_y", "_z"]:
                augmented_dict[name + suffix] = value  # Copy data for flipped images
    return augmented_dict

# --- PROCESS IMAGES & AUGMENT DATASET ---
batch2_names = set(curves_batch2.keys())  # Get original names for batch 2
batch3_names = set(curves_batch3.keys())  # Get original names for batch 3
# batch4_names = set(curves_batch4.keys())  # Get original names for batch 4

new_names_batch2 = []
new_names_batch3 = []
new_names_batch4 = []

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        base_name, flipped_names = flip_and_save(image_path, output_folder)
        
        # Determine if this name belongs to batch 2 or batch 3
        if base_name in batch2_names:
            new_names_batch2.extend(flipped_names)
        elif base_name in batch3_names:
            new_names_batch3.extend(flipped_names)
        # elif base_name in batch4_names:
        #     new_names_batch4.extend(flipped_names)

# Augment each dataset separately
aug_curves_batch2 = augment_data(curves_batch2, batch2_names, new_names_batch2)
aug_curves_batch3 = augment_data(curves_batch3, batch3_names, new_names_batch3)
# aug_curves_batch4 = augment_data(curves_batch4, batch4_names, new_names_batch4)
aug_e_com_batch2 = augment_data(e_com_batch2, batch2_names, new_names_batch2)
aug_e_com_batch3 = augment_data(e_com_batch3, batch3_names, new_names_batch3)
# aug_e_com_batch4 = augment_data(e_com_batch4, batch4_names, new_names_batch4)

# --- SAVE EXPANDED DATASET ---
np.savez(output_npz_file, 
         curves_batch2=aug_curves_batch2, 
         curves_batch3=aug_curves_batch3, 
        #  curves_batch4=aug_curves_batch4, 
         e_com_batch2=aug_e_com_batch2,
         e_com_batch3=aug_e_com_batch3
        # , e_com_batch4=aug_e_com_batch4
         )

print(f"✅ Dataset expansion complete! All original & flipped images saved in '{output_folder}', data saved in '{output_npz_file}'.")



#%% Plot Artificial Data to ensure correct (By manual comparrison to plots from Extract_curves_and_E_c)

# Validation essentially

# Load saved data and plot some curves for verification
data = np.load("processed_data_expanded.npz", allow_pickle=True)
curves_batch2 = data["curves_batch2"].item()
curves_batch3 = data["curves_batch3"].item()
# curves_batch4 = data["curves_batch4"].item()
e_com_batch2 = data["e_com_batch2"].item()
e_com_batch3 = data["e_com_batch3"].item()
# e_com_batch4 = data["e_com_batch4"].item()

# Plot some random force-strain curves
plt.figure(figsize=(10, 5))
for i, (name, (strain, force)) in enumerate(curves_batch2.items()):
    if name.endswith("_x"):
        plt.plot(strain, force, label=name)
plt.xlabel("Compression Strain [-]")
plt.ylabel("Stress Stress [MPa]")
plt.title("Artificial Sample Stress-Strain Curves (Batch 2)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
for i, (name, (strain, force)) in enumerate(curves_batch3.items()):
    if name.endswith("_x"):
        plt.plot(strain, force, label=name)
plt.xlabel("Compression Strain [-]")
plt.ylabel("Stress Applied [MPa]")
plt.title("Artificial Sample Stress-Strain Curves (Batch 3)")
plt.legend()
plt.grid()
plt.show()

# plt.figure(figsize=(10, 5))
# for i, (name, (strain, force)) in enumerate(curves_batch4.items()):
#     if name.endswith("_x"):
#         plt.plot(strain, force, label=name)
# plt.xlabel("Compression Strain [-]")
# plt.ylabel("Stress Applied [MPa]")
# plt.title("Artificial Sample Stress-Strain Curves (Batch 4)")
# plt.legend()
# plt.grid()
# plt.show()

print(e_com_batch2)
print(e_com_batch3)
# print(e_com_batch4)

print("Data extraction and verification complete!")