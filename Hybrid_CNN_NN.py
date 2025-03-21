#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#%% ---------------------------------
# 🔹 CNN Feature Extractor
# ---------------------------------
feature_nodes = 10
# Assign material and size properties: (UTS[MPa], Elongation at braek [%], Width or Length (should be equal) [mm], Height [mm])
# All values are normalized by max used value used in all data:
norm_UTS, norm_Elong, norm_WL, norm_H = 48, 160, 30, 30

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_features=feature_nodes):  # Reduced output features
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=1),  # Fewer filters
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 8, kernel_size=2, stride=1, padding=1),  
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(16 * 2 * 2, output_features) # Numbers need to line up with numbers defining input size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # extracted features

# ---------------------------------
# 🔹 Full Hybrid Model (CNN + MLP)
# ---------------------------------
class HybridCNN_NN(nn.Module):
    def __init__(self, num_features=feature_nodes, num_material_size_props=4, num_outputs=33):
        super(HybridCNN_NN, self).__init__()
        self.cnn = CNNFeatureExtractor(output_features=num_features)

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features + num_material_size_props, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, image, material_size_props):
        cnn_features = self.cnn(image)
        combined_input = torch.cat((cnn_features, material_size_props), dim=1)
        output = self.fc_layers(combined_input)
        return output[:, :-1], output[:, -1]  # (Stress, E_c)
    
# ---------------------------------
# 🔹 Custom Dataset Loader
# ---------------------------------
# 
class LatticeDataset(Dataset):
    '''
    This class should be customized based on the exact data and format used for training. This is heavily customized for my data.
    '''
    def __init__(self, image_folder, data_file, stress_norm):
        self.image_folder = image_folder
        self.data = np.load(data_file, allow_pickle=True)
        self.image_names = list(self.data["curves_batch2"].item().keys()) + list(self.data["curves_batch3"].item().keys())# + list(self.data["curves_batch4"].item().keys())
        self.curves = {**self.data["curves_batch2"].item(), **self.data["curves_batch3"].item()}#, **self.data["curves_batch4"].item()}
        self.e_com = {**self.data["e_com_batch2"].item(), **self.data["e_com_batch3"].item()}#, **self.data["e_com_batch4"].item()}

        self.stress_norm = stress_norm # Max stress in MPa (Set based on dataset)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, image_name + ".png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Image {img_path} not found")

        image = self.transform(image)

        strain, stress = self.curves[image_name]
        e_c = self.e_com.get(image_name, 0) / self.stress_norm

        # Normalize stress values
        stress = stress / self.stress_norm

        if image_name.startswith("b2"):
            material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 20/norm_WL, 20/norm_H])
        elif image_name.startswith("b3"):
            material_size_props = torch.tensor([3.23/norm_UTS, 160/norm_Elong, 25/norm_WL, 30/norm_H])  
        elif image_name.startswith("b4"):
            if image_name.endswith("v1r1") or image_name.endswith("v1r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 22/norm_WL, 22/norm_H])
            if image_name.endswith("v2r1") or image_name.endswith("v2r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 24/norm_WL, 24/norm_H])
            if image_name.endswith("v3r1") or image_name.endswith("v3r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 22/norm_WL, 22/norm_H])
            if image_name.endswith("v4r1") or image_name.endswith("v4r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 24/norm_WL, 24/norm_H])
            if image_name.endswith("v5r1") or image_name.endswith("v5r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 26/norm_WL, 26/norm_H])
            if image_name.endswith("v6r1") or image_name.endswith("v6r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 28/norm_WL, 28/norm_H])
            if image_name.endswith("v7r1") or image_name.endswith("v7r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 28/norm_WL, 28/norm_H])
            if image_name.endswith("v8r1") or image_name.endswith("v8r2"):
                material_size_props = torch.tensor([48/norm_UTS, 12/norm_Elong, 30/norm_WL, 30/norm_H])

        return image, material_size_props.float(), torch.tensor(stress, dtype=torch.float32), torch.tensor(e_c, dtype=torch.float32)
# ---------------------------------
# 🔹 Training Function
# ---------------------------------
def custom_loss(pred, true):
    '''
    Custom loss function, defined s´due to the uncentanty of the results.
    With this function, the error of a point will be 0 if it is within X of the true value
    '''
    loss_cutoff = 0.05 # if within 5% of true, give 0 error
    rel_diff = abs(true-pred)/true
    indices = rel_diff > loss_cutoff
    mseloss = torch.mean((pred[indices]-true[indices])**2)
    return mseloss

def train_model(model, dataloader, curve_weight, epochs=100, lr=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_track = []  # Initialize as an empty list

    for epoch in range(epochs):
        total_loss = 0
        for images, material_size_props, curves, e_c in dataloader:
            optimizer.zero_grad()
            pred_curves, pred_ec = model(images, material_size_props)
            loss = custom_loss(pred_curves, curves) * curve_weight + custom_loss(pred_ec, e_c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        loss_track.append(total_loss)  # append loss for later plotting
        print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

    return loss_track

# ---------------------------------
# 🔹 Running the Training
# ---------------------------------
image_folder = "All_pre_thesis_png_expanded" #"CNN_training_png_expanded"
data_file = "processed_data_expanded.npz"

stress_norm = 200 #MPa
dataset = LatticeDataset(image_folder, data_file, stress_norm=stress_norm)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = HybridCNN_NN()
loss_track = train_model(model, dataloader, curve_weight = 10, epochs = 30, lr = 0.0001)

plt.figure()
plt.plot(loss_track)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training Loss for Hybrid Network')

#Save model to utilize in RL environment
torch.save(model.state_dict(), "hybrid_nn_model.pth")


#%% ---------------------------------
# 🔹 Prediction on New Images (Optimized)
# ---------------------------------
new_image_folder = "png_images"  # Folder containing new PNGs

# Load new images
new_images = []
new_material_size_props = []
image_names = []

for img_name in os.listdir(new_image_folder):
    if img_name.endswith(".png"):
        img_path = os.path.join(new_image_folder, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Image {img_name} not found.")
            continue

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize & add channel dim
        new_images.append(image)
        new_material_size_props.append(torch.tensor([48/norm_UTS, 12/norm_Elong, 20/norm_WL, 20/norm_H]))  # UTS[MPa], Elongation at break[%]
        image_names.append(img_name)

# Convert to batch tensors
new_images = torch.stack(new_images)
new_material_size_props = torch.stack(new_material_size_props)

# Run inference
model.eval()
with torch.no_grad():
    pred_curves, pred_ec = model(new_images, new_material_size_props)

# Plot Predictions
# Denormalization before plotting

for i in range(len(image_names)):
    plt.figure(figsize=(6, 4))
    strain_plot = np.linspace(0, 0.3, 32)  # Defined in training data

    # Convert normalized values back to real stress values
    denormalized_stress = pred_curves[i].cpu().numpy() * stress_norm

    plt.plot(strain_plot, denormalized_stress, label="Predicted Stress-Strain Curve", color="blue")
    plt.xlabel("Strain")
    plt.ylabel("Predicted Stress (MPa)")
    plt.title(f"Prediction for {image_names[i]}\nE_compression: {pred_ec[i].item()*stress_norm:.4f} MPa")
    plt.legend()
    plt.grid()
    plt.show()
