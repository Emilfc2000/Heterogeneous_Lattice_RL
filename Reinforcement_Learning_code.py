import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import trimesh
import time
from skimage.draw import polygon
from PIL import Image
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from Hybrid_CNN_NN import model as CNN_model

# Load the pre-trained CNN model
CNN_model.eval()
for param in CNN_model.parameters():
    param.requires_grad = False  # Freeze parameters to prevent accidental training

#Define file locations and names
RL_folder = r"RL_training_folder/" #Folder to keep all intermediate files for training (json, stl, png)
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  # nTopCL path
nTopFilePath = "lattice_auto.ntop"  # Path to your nTop file

#Geometry of desired design:
Length, Width, Height = 20, 20, 20 #mm, mm, mm
# When making json file, the width is set to 1mm, in order to optimize ntop meshing time.

class LatticeEnv(Env):
    def __init__(self):
        super(LatticeEnv, self).__init__()
        
        # Define specific low and high limits for each of the design parameters:
        # Beam Thickness (mm), Cell Count, Length, Width, Height (These 3 params should have same value)
        self.action_low_limits = np.array([0.4, 140, Length, Width, Height], dtype=np.float32)
        self.action_high_limits = np.array([0.8, 230, Length, Width, Height], dtype=np.float32)

        # Update action space with different bounds per parameter
        self.action_space = spaces.Box(low=self.action_low_limits, high=self.action_high_limits, dtype=np.float32)

        # Define observation space - "Area of interest"
        self.obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_high = np.array([200.0, 0.3, 200.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        
        self.current_step = 0
        self.count = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.state = np.array([Length, Width, Height])  # Placeholder state
        
        return self.state, {}
    
    def step(self, action):
        start_time = time.time()
        self.current_step += 1
        
        # Generate lattice stl structure using nTop command line and json input files
        filename, stl_path = self.generate_lattice(action)

        # Convert stl of lattice to png
        image_path = self.stl_to_png(filename=filename, stl_path=stl_path)

        # Load image and preprocess for CNN
        img = Image.open(image_path).convert('L') #.convert('RGB')
        img = self.preprocess_image(img)

        # Predict Stress-Strain and E_c using CNN
        strain, stress, E_c = self.predict(img)
        stress = max(stress)

        obs1 = max(stress).item()
        obs2 = strain[np.argmax(stress)]
        obs3 = E_c.item()
        obs = np.array([obs1, obs2, obs3], dtype = np.float32)

        # Reward function
        reward = self.reward_calc(stress=stress, E_c=E_c, action=action)
        
        done = True # Define episode termination condition
        truncated = False
        
        return obs, reward, done, truncated, {}
    
    def generate_lattice(self, action):
        # Generate Json file form action
        # run json through ntop
        # Name file according to self.count
        self.count += 1
        print("Generating Lattice Structure "+str(self.count)+"...")
        input_filename = os.path.join(RL_folder, f"RL_input_{self.count}.json")
        output_filename = os.path.join(RL_folder, f"RL_output_{self.count}.json")

        Optimize_time_width = float(2) # Used to reduce time of mesh generation in ntop
        # If use original size, exchange with float(action[3])

        # Construct input JSON
        input_json = {
        "inputs": [
            {"name": "Beam Thickness", "type": "scalar", "values": float(action[0]), "units": "mm"},
            {"name": "Voronoi Cell Count", "type": "integer", "values": int(action[1])},
            {"name": "Random Seed", "type": "integer", "values": 1},
            {"name": "Name", "type": "text", "value": str(self.count)},
            {"name": "Length", "type": "scalar", "values": float(action[2]), "units": "mm"},
            {"name": "Width", "type": "scalar", "values": Optimize_time_width, "units": "mm"},
            {"name": "Height", "type": "scalar", "values": float(action[4]), "units": "mm"},
            {"name": "Save Folder Path", "type": "text", "value": RL_folder}
            ]
        }

        # Save JSON file
        with open(input_filename, 'w') as f:
            json.dump(input_json, f, indent=4)

        # Construct nTopCL command
        command = [exePath, "-j", input_filename, "-o", output_filename, "-v1", nTopFilePath]

        # Run nTopCL
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        filename = str(self.count)+".stl"
        stl_path = os.path.join(RL_folder, filename)

        return filename, stl_path
    
    def stl_to_png(self, filename, stl_path):
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
        output_image_path = os.path.join(RL_folder, f"{os.path.splitext(filename)[0]}.png")
        plt.imsave(output_image_path, binary_image, cmap="gray")

        return output_image_path

    def preprocess_image(self, img):
        # Preprocess image for CNN input - ensure same format as trianing images
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        return transform(img)

    def predict(self, img):
        norm_UTS, norm_Elong, norm_WL, norm_H = 48, 160, 30, 30
        m_s_p = torch.tensor([48/norm_UTS, 12/norm_Elong, 20/norm_WL, 20/norm_H]).unsqueeze(1).reshape(1,-1)
        with torch.no_grad():
            output = CNN_model(img.unsqueeze(0), material_size_props=m_s_p) * 200 #normilzed by 200MPa in training. This number is defined in line 167 of Hybrid_CNN_NN.py
        stress_curve, e_c = output[0], output[1]
        strain_curve = np.linspace(0,0.3,32)
        #Stress is 32 points in which correspond to stress values at np.linspace(0.0, 0.3, 32) strain values
        return strain_curve, stress_curve, e_c

    def reward_calc(self, stress, E_c, action):
        #Change reward function to alter RL optimizaion
        bt = action[0] # Beam thickness
        cc = action[1] # Voronoi Cell Count
        reward = E_c.item()*2 + max(stress).item()*5 - bt*20 - cc/20
        return float(reward)


# Callback to track rewards during training
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []
        self.num_envs = 0

    def _on_training_start(self) -> None:
        self.num_envs = self.training_env.num_envs
        self.current_rewards = [0.0] * self.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        
        for i in range(self.num_envs):
            self.current_rewards[i] += rewards[i]
            if dones[i]:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0.0
        
        return True
    
# class RewardLoggerCallback(BaseCallback):
#     def _on_step(self) -> bool:
#         rewards = self.locals["rewards"]
#         if isinstance(rewards, np.ndarray):
#             rewards = rewards.flatten()  # Ensure it’s a 1D array
#         self.episode_rewards.extend(rewards.tolist())
#         return True

reward_logger = RewardLoggerCallback(verbose=1)

#%% Train the RL model

# Possibility to run multiple environments at once - shouldn't be done with file generationg
# num_envs = 1
# env = SubprocVecEnv([lambda: LatticeEnv() for _ in range(num_envs)])
# env = VecMonitor(env)

env = DummyVecEnv([lambda: LatticeEnv()])
env = VecMonitor(env)

RL_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_lattice_tensorboard/", batch_size=4, n_steps=4, device="cpu")
RL_model.learn(total_timesteps=20, callback=reward_logger) #total_timesteps is number of episodes

RL_model.save("ppo_lattice_model")

# Plot reward progression
plt.plot(reward_logger.episode_rewards, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Progression During Training")
plt.legend()
plt.show()

#%% Use model to generate a structure:

env = LatticeEnv()
state, _ = env.reset()

action, _states = RL_model.predict(state, deterministic=True)
print(f"Optimal action for a {action[2]}mm x {action[3]}mm x {action[4]}mm (Length x Width x Height) design space is as such:")
print("Beam Thickness = " + str(action[0]))
print("Cell count = " + str(action[1]))
