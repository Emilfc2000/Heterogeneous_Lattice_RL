import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


#Define file locations and names
# RL_folder = r"C:/Users/emilf/OneDrive - Aarhus universitet/Uni/10. Semester/Codes_and_files/RL_training_folder/"
RL_folder = r"RL_training_folder/" # Folder to keep all intermediate files for training (json, csv, stl, png)
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  # nTopCL path
nTopFilePath = "lattice_auto_v3.ntop"  # Path to your nTop file

#Geometry of desired design:
Length_min, Width_min, Height_min = 20, 20, 20 #mm, mm, mm
# Different options for max dimension:
Length_max, Width_max, Height_max = Length_min, Width_min, Height_min #mm, mm, mm
# Length_max, Width_max, Height_max = 25, 25, 30 #mm, mm, mm

class LatticeEnv(Env):
    def __init__(self):
        super(LatticeEnv, self).__init__()
        
        # Define specific low and high limits for each of the design parameters:
        # Beam Thickness (mm), Cell Count, Length, Width, Height
        a_low_params = np.array([0.4, 120, Length_min, Width_min, Height_min], dtype=np.float32)
        a_high_params = np.array([0.8, 230, Length_max, Width_max, Height_max], dtype=np.float32)

        # Adding the 100 control actions to control the density point map distrubution for the lattice in nTop
        a_low_dens = np.ones(100)
        a_high_dens = np.ones(100)*10

        #Combining them for full action limits:
        self.action_low_limits = np.concatenate((a_low_params, a_low_dens), dtype=np.float32)
        self.action_high_limits = np.concatenate((a_high_params, a_high_dens), dtype=np.float32)

        # Update action space with different bounds per parameter
        self.action_space = spaces.Box(low=self.action_low_limits, high=self.action_high_limits, dtype=np.float32)

        # Define observation space - "Area of interest"
        # obs1: Stress_var[MPa^2], obs2: stress_max[MPa], obs3: energy_abs[J], obs4: E_c[MPa]
        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_high = np.array([1.0e5, 1.0e3, 10.0, 1.0e3], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        
        # Define Compressions and Loads for simulations:
        self.Compressions = (-np.array([0.03, 0.06, 0.09, 0.12, 0.15])).tolist()
        self.Loads = (-np.array([30, 60, 90, 120, 150])).tolist()

        # Material properties: [Density [kg/m^3], Poisson's Ratio [], Youngs mod [MPa], UTS [MPa]]
        self.blackV4 = [1200, 0.35, 2800, 65] # https://formlabs.com/eu/store/materials/black-resin-v4/?srsltid=AfmBOooV6wkFh0Tjvj68ALg3bF4jgPiMXTK_qsLtSnzcyVVrIkFpAGt7
        self.Elastic50a = [1200, 0.45, 10, 3.23] # https://formlabs-media.formlabs.com/datasheets/2001420-TDS-ENUS-0.pdf

        self.current_step = 0
        self.count = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reseting step
        self.current_step = 0

        # Initializing with a random intial state.
        # The state doesn't contribute to reward function, but it might influence first action
        random_initial_state = np.random.rand(4)
        self.state = random_initial_state*self.obs_high
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1

        # Generate density field point map as csv file for nTop to import in next part of step:
        self.generate_density_field(action)
        
        # Generate lattice structure using nTop command line and JSON input files
        rel_density = self.generate_lattice(action)

        # Importing ntop simulations data:
        # (Variance of stress field, Maximums of stress fields,
        # Absorbed energy, Compressibe Elastic Modulus)
        s_vars, s_maxs, E_abs, E_c = self.ntop_sims(action)

        # Defining observations to return back to agent
        obs1 = max(s_vars)
        obs2 = max(s_maxs)
        obs3 = E_abs
        obs4 = E_c
        obs = np.array([obs1, obs2, obs3, obs4], dtype = np.float32)

        # Reward function
        reward = self.reward_calc(s_vars=s_vars, s_maxs=s_maxs, E_abs=E_abs, E_c=E_c, rel_density=rel_density, action=action)
        
        done = True # Such that each training episode consists of only ONE action
        truncated = False
        
        return obs, reward, done, truncated, {}

    def generate_density_field(self, action):
        # Will generate a field over the lattice.
        # Utilizees lattice dimensions and center, as well as density values form action
        L, W, H = action[2:5] # Dimensions of lattice
        dens_vals = action[5:] # All 100 density values
        c_x, c_y, c_z = 10, 0, 10 # mm center of lattice face - latticing face is on x-z plane, so y = 0

        N = len(dens_vals)
        N_x = 10
        N_y = int(np.floor(N/N_x))
        # To make robust (In case one inputs other than 100 inputs, this will cause it not to break)
        # Will not enter this if statement if Agent gives 100 dens actions as intiall/currently defined.
        if N%N_x != 0:
            n_remove = N - N_x*N_y
            dens_vals = dens_vals[:-n_remove] # Removes last n_remove number of action values.
            N = len(dens_vals)

        # Defining the grid based on dimensions
        x_ = np.linspace(c_x-L/2, c_x+L/2, N_x).reshape(1,-1)
        z_ = np.linspace(c_z-H/2, c_z+H/2, N_y).reshape(-1,1)
        xz_mesh = np.meshgrid(x_,z_)

        output = np.zeros((N + 1, 4)) # Output is x,y,z,dens (for each grid point)

        # Field needs a point outside of x-z plane to interpolate properly in nTop.
        # y=10 so far enough away to not actually effect lattice density field on the plane.
        output[0,:] = [10,10,10,1] # Some random point far enough away to not effect, but to make it 3D
        output[1:,0] = xz_mesh[0].reshape(-1,1).squeeze(1)
        output[1:,1] = np.zeros(N) # Values are already 0 before, but for good measure
        output[1:,2] = xz_mesh[1].reshape(-1,1).squeeze(1)
        output[1:,3] = dens_vals

        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(output)

        # Save DataFrame to .csv
        df.to_csv(r'RL_training_folder/ramp_input.csv', index=False)

        return

    def generate_lattice(self, action):
        # Generate Json file form action
        # run json through ntop
        # Name file according to self.count
        self.count += 1
        print("Generating & Simulating Lattice Structure "+str(self.count)+"...")
        input_filename = os.path.join(RL_folder, f"RL_input.json")
        output_filename = os.path.join(RL_folder, f"RL_output.json")

        Optimize_time_width = float(2) # Used to reduce time of mesh generation in ntop
        # If use original size, exchange with float(action[3])

        # Construct input JSON
        input_json = {
        "inputs": [
            {"name": "Beam Thickness", "type": "scalar", "values": float(action[0]), "units": "mm"},
            {"name": "Voronoi Cell Count", "type": "integer", "values": int(action[1])},
            {"name": "Random Seed", "type": "integer", "values": 1},
            {"name": "Length", "type": "scalar", "values": float(action[2]), "units": "mm"},
            {"name": "Width", "type": "scalar", "values": float(action[3]), "units": "mm"},
            # {"name": "Width", "type": "scalar", "values": Optimize_time_width, "units": "mm"},
            {"name": "Height", "type": "scalar", "values": float(action[4]), "units": "mm"},
            {"name": "Save Folder Path", "type": "text", "value": RL_folder},
            {"name": "Compression List", "type": "scalar_list", "value": self.Compressions},
            {"name": "Force List", "type": "scalar_list", "value": self.Loads, "units": "N"}            
            ]
        }

        # Save JSON file
        with open(input_filename, 'w') as f:
            json.dump(input_json, f, indent=4)

        # Construct nTopCL command
        command = [exePath, "-j", input_filename, "-o", output_filename, "-v1", nTopFilePath]
        # This also, through nTop, runs a series of simulations of compression testing, and generates a stress and displacement field csv files

        # Run nTopCL
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Extract relative density "val" from the output JSON
        with open(output_filename, 'r') as f:
            output_data = json.load(f)
            val = output_data[0]["value"]["val"]
            rel_density = round(val, 3)  # Round to 3 significant digits

        return rel_density

    def get_U_z(self,data,L,W,H):
        # Compute distances to center of load face
        data["distance"] = np.sqrt((data["X [mm]"] - L/2)**2 + 
                                   (data["Y [mm]"] - W/2)**2 + 
                                   (data["Z [mm]"] - H)**2)

        # Find the row with the minimum distance - to take compression at center of load face
        closest_row = data.loc[data["distance"].idxmin()]

        # Get the corresponding U_z value
        U_z = closest_row["U_z [m]"]
        return U_z

    def ntop_sims(self, action):
        # Function to import and analyze the FEA simulation data from nTop.
        L, W, H = action[2], action[3], action[4] #Length Width and Height of current lattice structure

        # Stress Fields - Import from nTop and utilize:
        N_stress_sims = 5 # Number of Specified Compression simulations in nTop file
        stress_vars = np.zeros(N_stress_sims) # Initialize array for data
        stress_maxs = np.zeros(N_stress_sims) # Initialize array for data
        for i in range(1, 1+N_stress_sims):
            stress_file = RL_folder+f"stress_{i}.csv"
            data = pd.read_csv(stress_file, header=None)
            data.columns = ["X [mm]","Y [mm]","Z [mm]","Stress [Pa]"]
            
            # Obtaining variance of stress field at each compression %
            stress_vars[i-1] = np.var(data["Stress [Pa]"]/1e6) # unit: MPa^2

            # Obtaining max local stress on structure at each compression %
            stress_maxs[i-1] = max(data["Stress [Pa]"]/1e6) # unit: MPa

        # Displacement Fields - Import from nTop and utilize:
        N_disp_sims = 5 # Number of Specified Load simulations in nTop file
        U_z = np.zeros(N_disp_sims) # Initialize array for data
        for i in range(1, 1+N_disp_sims):
            disp_file = RL_folder+f"displacement_{i}.csv"
            data = pd.read_csv(disp_file, header=None)
            data.columns = ["X [mm]", "Y [mm]", "Z [mm]", "U_x [m]", "U_y [m]", "U_z [m]"]

            # Get displacement values for each compression step
            U_z[i-1] = self.get_U_z(data, L, W, H)
        
        # Make arrays to integrate Force-displacement courve
        y = np.concatenate((np.array([0]), -np.asarray(self.Loads)))
        x = np.concatenate((np.array([0]), -U_z),axis=0)
        Energy_absorbed = simpson(y=y, x=x) #(y,x) input for some reasnm, unit: N*m=J

        strain = x*1e3/H # unitless (Compression strain is usually designated as negative [-])
        stress = y/(L*W) # unit: N/mm^2 = MPa
        gradiants = (stress[1:]-stress[:-1])/(strain[1:]-strain[:-1])
        E_c= np.mean(gradiants) #Estimated Compressive Elasticity modulus, unit: MPa

        return stress_vars, stress_maxs, Energy_absorbed, E_c

    def reward_calc(self, s_vars, s_maxs, E_abs, E_c, rel_density, action):
        # Calculates the reward for the RL training
        # To optimize training, this function/weights should be tuned as desired - significant time

        # Small algorithm to consider stress variance at different compression stages
        def sv_algorithm(s_vars):
            sv = 0
            for i in range(len(s_vars)):
                sv += s_vars[i] * 1/(100**((i+1)/5))
            return sv

        # All components of reward calculations, with weigths - these weights should be tuned
        # Positive components (The higher the better)
        Ec = E_c * 1/50 # Compressive E [MPa] * weight
        Ea = E_abs * 30 # Energy Absorbed [J] * weight
        # Negative components (The lower the better)
        sv = sv_algorithm(s_vars) * 1/20 # Variance of stress fields [MPa^2] * weight
        sm = sum(s_maxs>self.Elastic50a[3]) * 10 # When max stress [MPa] is above UTS * weight
        bt = action[0] * 15 # Beam thickness [mm] * weight
        cc = action[1] * 1/25 # Voronoi Cell Count [] * weight
        rd = rel_density * 10 # Relative Density [] * weight
        Lw = action[2] * 1/3 # Length of lattice [mm] * weight
        Ww = action[3] * 1/3 # Width of lattice [mm] * weight
        Hw = action[4] * 1/3 # Height of lattice [mm] * weight
        dm = np.mean(action[5:] - 5.5) * 20 # Mean Error on Density field * weight. Cannot do squared or absolute error

        # Manualyl change and tailor this function to optimize for desired values
        # reward = Ec + Ea - sv - sm - bt - cc - rd - Lw - Ww - Hw - dm
        reward = Ea - rd - dm

        return float(reward)


# Callback to track rewards during training. DON'T CHANGE - necessary for RL wo tork properly
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
    
reward_logger = RewardLoggerCallback(verbose=1)

#%% Train the RL model
env = DummyVecEnv([lambda: LatticeEnv()])
env = VecMonitor(env)

#Defining the PPO agent (A2C - Advantage Actor Critic agent with Proximal Policy Optimization of a Multi-Layer Perceptron NN Policy)
RL_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_lattice_tensorboard/", batch_size=2, n_steps=2, device="cpu")
RL_model.learn(total_timesteps=10, callback=reward_logger) #total_timesteps is number of episodes

RL_model.save("ppo_lattice_model")

# Plot reward progression
plt.plot(reward_logger.episode_rewards, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Progression During Training")
plt.legend()
plt.show()

#%% Use RL model to predict optimal BT and CC:

# Re-initialize environment
env = LatticeEnv()
state, _ = env.reset()

# Predict using RL_model
action, _states = RL_model.predict(state, deterministic=True)
print(f"Optimal action for a {action[2]}mm x {action[3]}mm x {action[4]}mm (Length x Width x Height) design space is as such:")
print(f"Beam Thickness:                     {action[0]:.4g}")
print(f"Cell count:                         {action[1]:.0f}")

# One can now run these through ntop to generate the optimal lattice structure