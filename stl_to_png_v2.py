import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.draw import polygon

# Input and output folders
input_folder = "stl_files"  # Replace with the folder containing STL files
output_folder = "png_images"  # Replace with the folder where to save the PNG images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all STL files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".stl"):
        stl_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        
        # Load the STL file
        mesh = trimesh.load_mesh(stl_path)

        # Project the mesh onto the X-Z plane to create a 2D cross-section
        vertices = mesh.vertices[:, [0, 2]]  # Extract X and Z coordinates

        # Normalize the coordinates to fit within a 512x512 image
        image_size = (512, 512)
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        z_min, z_max = vertices[:, 1].min(), vertices[:, 1].max()
        vertices[:, 0] = (vertices[:, 0] - x_min) / (x_max - x_min) * (image_size[0] - 1)
        vertices[:, 1] = (vertices[:, 1] - z_min) / (z_max - z_min) * (image_size[1] - 1)

        # Create a blank binary image
        binary_image = np.zeros(image_size, dtype=np.uint8)

        # Get the convex hull or triangulated faces to define filled regions
        if isinstance(mesh, trimesh.Trimesh):
            for face in mesh.faces:
                x = vertices[face, 0]
                y = vertices[face, 1]
                rr, cc = polygon(y, x, binary_image.shape)  # Fill polygon
                binary_image[rr, cc] = 255  # Set solid region to white

        # Save the binary image as a PNG using matplotlib
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
        plt.imsave(output_image_path, binary_image, cmap="gray")
        print(f"Saved {output_image_path}")

# Display the last image as a preview
plt.imshow(binary_image, cmap="gray")
plt.title("Last Processed 2D Cross-Section (X-Z Plane)")
plt.axis("off")
plt.show()
