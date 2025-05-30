import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import time
from scipy.integrate import simpson
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


#Define file locations and names
# RL_folder = r"C:/Users/emilf/OneDrive - Aarhus universitet/Uni/10. Semester/Codes_and_files/RL_training_folder/"
RL_folder = "RL_training_folder\\" # Folder to keep all intermediate files for training (json, csv, stl, png)
This_folder = os.getcwd()
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  # nTopCL path
nTopFilePath = "lattice_auto_v3.ntop"  # Path to your nTop file

#Geometry of desired design:
Length_min, Width_min, Height_min = 20, 5, 20 #mm, mm, mm
# Different options for max dimension:
Length_max, Width_max, Height_max = Length_min, Width_min, Height_min #mm, mm, mm
# Length_max, Width_max, Height_max = 25, 25, 30 #mm, mm, mm

# Number of density control points: (Should be a square: 4, 9, 16, 25, .... 100)
N_control = 25
max_distance = 3 # Maximum avereage relative distance between cell seeds (relative to the minimum; 1)

class LatticeEnv(Env):
    def __init__(self):
        super(LatticeEnv, self).__init__()
        # Define specific low and high limits for each of the design parameters:
        # Beam Thickness (mm), Cell Count, Length, Width, Height
        self.a_low_params = np.array([0.5, 120, Length_min, Width_min, Height_min], dtype=np.float32)
        # self.a_high_params = np.array([0.7, 150, Length_max, Width_max, Height_max], dtype=np.float32)
        self.a_high_params = np.array([0.7, 140, Length_max, Width_max, Height_max], dtype=np.float32)

        # Adding the control actions to control the density point map distrubution for the lattice in nTop
        self.a_low_dens = np.ones(N_control)
        self.a_high_dens = np.ones(N_control)*max_distance

        #Combining them for full action limits:
        self.action_low_limits = np.concatenate((self.a_low_params, self.a_low_dens), dtype=np.float32)
        self.action_high_limits = np.concatenate((self.a_high_params, self.a_high_dens), dtype=np.float32)
        # These upper and lower limits will be used to de-normalize actions via denormalization_action function
        
        # Normalized Actions (To ensure good exploring)
        self.act_norm_low = -1*np.ones(len(self.action_low_limits))
        self.act_norm_high = np.ones(len(self.action_high_limits))
        # Update action space with normalized action spaces. This should optimize training exploring
        self.action_space = spaces.Box(low=self.act_norm_low, high=self.act_norm_high, dtype=np.float32)

        # Define observation space - "Area of interest"
        # obs1: Stress_var[MPa^2], obs2: stress_max[MPa], obs3: energy_abs[J], obs4: E_c[MPa]
        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_high = np.array([1.0e5, 1.0e3, 10.0, 1.0e3], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        
        # Define Compressions and Loads for simulations:
        self.Compressions = (-np.array([0.025, 0.050, 0.075, 0.10, 0.125])).tolist() # Compressive strain
        # self.Loads = (-np.array([120, 240, 360])).tolist() # Newton - For blackV4 material
        self.Loads = (-np.array([4, 8, 12])).tolist() # Newton - For Elastic50a material

        # Material properties: [Density [kg/m^3], Poisson's Ratio [], Youngs mod [MPa], UTS [MPa]]
        self.blackV4 = [1200, 0.35, 2800, 65] # https://formlabs.com/eu/store/materials/black-resin-v4/?srsltid=AfmBOooV6wkFh0Tjvj68ALg3bF4jgPiMXTK_qsLtSnzcyVVrIkFpAGt7
        self.Elastic50a = [1200, 0.45, 1, 3.23] # https://formlabs-media.formlabs.com/datasheets/2001420-TDS-ENUS-0.pdf

        self.current_step = 0
        self.count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reseting step
        self.current_step = 0

        # Initializing with zeros, as state technically shouln't have any impact on actions ideally
        self.state = np.zeros(4)
        
        return self.state, {}
    
    def step(self, action):
        # Obligatory step function within the environment, which the RL library will interact with.
        # Within this function, everything that should be done during each step should be included.
        self.current_step += 1
        action = self.denormalize_action(action)
        print(f"Actions [bt, cc, L, W, H]:\n{action[:5]}")
        print(f"Density field:\n{action[5:10]}\n{action[10:15]}\n{action[15:20]}\n{action[20:25]}\n{action[25:30]}")

        # Generate density field point map as csv file for nTop to import in next part of step:
        self.generate_density_field(action)
        
        # Generate lattice structure using nTop command line and JSON input files
        rel_density = self.generate_lattice(action)

        # In case Simulations fails, rel_density = 0, and ignore all code and 
        if rel_density == 0:
            reward = reward_calc_failed()
        else:
            # Importing ntop simulations data:
            # (Variance of stress field, Maximums of stress fields,
            # Absorbed energy, Compressibe Elastic Modulus)
            s_vars, s_maxs, E_abs, E_c, npr, dir_deform = self.ntop_sims(action)

            # Reward function
            reward = self.reward_calc(s_vars=s_vars, s_maxs=s_maxs, E_abs=E_abs, E_c=E_c, npr=npr, dir_deform=dir_deform, rel_density=rel_density, action=action)
        
        print('Reward:', reward)
        # Delete the files, to ensure next epsiode utilizes new simulation data, and not accidentally the old
        filenames = ["stress_1.csv", "stress_2.csv","stress_3.csv", "stress_4.csv","stress_5.csv", "displacement_1.csv", "displacement_2.csv", "displacement_3.csv"]
        for name in filenames:
            file_path = RL_folder+name
            if os.path.exists(file_path):
                os.remove(file_path)

        # Defining observations to return back to agent - constant 0 to stabilize training i think
        obs = np.zeros(4)

        done = True # Such that each training episode consists of only ONE action
        truncated = False # Doesn't really matter because done always True

        return obs, reward, done, truncated, {}

    def denormalize_action(self, action):
        # Function made to denormalize the actions taken by the agent
        # Can be used outside of the class for validation, as shown at the bottom.
        # Normalizing the Action from -1,1 should increase efficiency and speedd of learning
        action = np.clip(action, self.act_norm_low, self.act_norm_high) # Ensure no breach of boundary
        true_actions = self.action_low_limits + (action + 1) / 2 * (self.action_high_limits - self.action_low_limits)
        true_actions[1] = int(true_actions[1])
        return true_actions.astype(np.float32)

    def generate_density_field(self, action):
        # Will generate a field over the lattice.
        # Utilizees lattice dimensions and center, as well as density values form action
        L, W, H = action[2:5] # Dimensions of lattice
        dens_vals = action[5:] # All N_control density values

        N = len(dens_vals)
        N_x = int(np.sqrt(N_control))
        N_y = int(np.floor(N/N_x))
        # To make code robust (In case one inputs other than N_control inputs, this will cause it not to break)
        # Will not enter this if statement if Agent gives N_control dens actions as intiall/currently defined.
        if N%N_x != 0:
            n_remove = N - N_x*N_y
            dens_vals = dens_vals[:-n_remove] # Removes last n_remove number of action values.
            N = len(dens_vals)

        # Defining the grid based on dimensions
        x_ = np.linspace(-L/2, L/2, N_x).reshape(1,-1)
        z_ = np.linspace(-H/2, H/2, N_y).reshape(-1,1)
        xz_mesh = np.meshgrid(x_,z_)

        output = np.zeros((N + 1, 4)) # Output is x,y,z,dens (for each grid point)

        # Field needs a point outside of x-z plane to interpolate properly in nTop.
        # y=10 so far enough away to not actually effect lattice density field on the plane.
        output[0,:] = [0,100,0,1] # Some random point far enough away to not effect, but to make it 3D
        output[1:,0] = xz_mesh[0].reshape(-1,1).squeeze(1)
        output[1:,1] = np.ones(N)*(-W/2) # On the front face
        output[1:,2] = xz_mesh[1].reshape(-1,1).squeeze(1)
        output[1:,3] = dens_vals

        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(output)

        # Save DataFrame to .csv
        df.to_csv(os.path.join(RL_folder,"ramp_input.csv"), index=False, header=False)

        return

    def generate_lattice(self, action):
        # Generate Json file form action
        # run json through ntop
        # Name file according to self.count
        self.count += 1
        print("Generating & Simulating Lattice Structure "+str(self.count)+"...")
        input_filename = os.path.join(RL_folder, f"RL_input.json")
        output_filename = os.path.join(RL_folder, f"RL_output.json")

        # Construct input JSON
        input_json = {
        "inputs": [
            {"name": "Beam Thickness", "type": "scalar", "values": float(action[0]), "units": "mm"},
            {"name": "Voronoi Cell Count", "type": "integer", "values": int(action[1])},
            {"name": "Random Seed", "type": "integer", "values": 1},
            {"name": "Length", "type": "scalar", "values": float(action[2]), "units": "mm"},
            {"name": "Width", "type": "scalar", "values": float(action[3]), "units": "mm"},
            {"name": "Height", "type": "scalar", "values": float(action[4]), "units": "mm"},
            {"name": "Save Folder Path", "type": "text", "value": This_folder+"\\"+RL_folder},
            {"name": "Compression List", "type": "scalar_list", "value": self.Compressions},
            {"name": "Force List", "type": "scalar_list", "value": self.Loads, "units": "N"},
            {"name": "Ramp File", "type": "text", "value": "ramp_input.csv"}          
            ]
        }

        # Save JSON file
        with open(input_filename, 'w') as f:
            json.dump(input_json, f, indent=4)

        # Construct nTopCL command
        command = [exePath, "-j", input_filename, "-o", output_filename, "-v1", nTopFilePath]
        # This also, through nTop, runs a series of simulations of compression testing, and generates a stress and displacement field csv files

        # Run nTopCL
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the output file was created
        filenames = ["stress_1.csv", "stress_2.csv","stress_3.csv", "stress_4.csv","stress_5.csv", "displacement_1.csv", "displacement_2.csv", "displacement_3.csv"]
        for name in filenames:
            testfile = os.path.join(RL_folder, name)
            if not os.path.exists(testfile):
                print("❌ Output file was not created.")
                print("Return code:", result.returncode)
                print("STDOUT:\n", result.stdout.strip())
                print("STDERR:\n", result.stderr.strip())

                # In case Simulation Fails
                rel_density = 0
                return rel_density
            
                # Raise FileNotFoundError to crash cleanly - this doesn't currently run, as function ends at return
                raise FileNotFoundError(f"nTopCL failed: output file '{testfile}' was not generated.")


        # Extract relative density "val" from the output JSON
        with open(output_filename, 'r') as f:
            output_data = json.load(f)
            val = output_data[0]["value"]["val"]
            rel_density = round(val, 3)  # Round to 3 significant digits

        return rel_density

    def get_U_z(self, data, L, W, H):
        # Compute distances to center of load face. Will be at (0, W/2, H/2)
        data["distance_ct"] = np.sqrt((data["X [mm]"] - 0)**2 +
                                      (data["Y [mm]"] - W/2)**2 +
                                      (data["Z [mm]"] - H/2)**2)

        # Find the row with the minimum distance - to take compression at center of load face
        closest_row = data.loc[data["distance_ct"].idxmin()]

        # Get the corresponding U_z value
        U_z = closest_row["U_z [m]"]
        return U_z
    
    def get_npr(self, data, L, W, H):
        # Function to obtain the constraint for npr - negative poisson ratio - optimization
        # Define region boundaries of intersting structure
        z_lower = -H / 4
        z_upper = H / 4
        x_right_min = L / 2 - 2
        x_left_max = - (L / 2 - 2)

        # Select points within the right region
        closest_rows_right = data[
            (data["X [mm]"] >= x_right_min) &
            (data["Z [mm]"] >= z_lower) &
            (data["Z [mm]"] <= z_upper)
        ]

        # Select points within the left region
        closest_rows_left = data[
            (data["X [mm]"] <= x_left_max) &
            (data["Z [mm]"] >= z_lower) &
            (data["Z [mm]"] <= z_upper)
        ]

        # Compute mean U_x values
        mean_U_x_right = closest_rows_right["U_x [m]"].mean()
        mean_U_x_left = closest_rows_left["U_x [m]"].mean()

        # Want "inward" movement to be good
        mean_U_x = np.mean([mean_U_x_left, -mean_U_x_right])

        return mean_U_x * 1e3 # To get mm unit

    def direction_deformation(self, data, L, W, H):
        # Select all lattice points
        z_lower = -H / 2
        z_upper = H / 2
        lattice_rows = data[
            (data["Z [mm]"] >= z_lower) &
            (data["Z [mm]"] <= z_upper)
        ]

        # Compute average U_x (positive direction) - enfore rightwards movement
        mean_U_x_right = lattice_rows["U_x [m]"].mean()

        return mean_U_x_right

    def ntop_sims(self, action):
        # Function to import and analyze the FEA simulation data from nTop.
        L, W, H = action[2], action[3], action[4] #Length Width and Height of current lattice structure

        # Stress Fields - Import from nTop and utilize:
        N_stress_sims = len(self.Compressions) # Number of Specified Compression simulations in nTop file
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
        N_disp_sims = len(self.Loads) # Number of Specified Load simulations in nTop file
        U_z = np.zeros(N_disp_sims) # Initialize array for data
        npr = np.zeros(N_disp_sims) # Initialize array for data
        dir_deform = np.zeros(N_disp_sims) # Initialize array for data
        for i in range(1, 1+N_disp_sims):
            disp_file = RL_folder+f"displacement_{i}.csv"
            data = pd.read_csv(disp_file, header=None)
            data.columns = ["X [mm]", "Y [mm]", "Z [mm]", "U_x [m]", "U_y [m]", "U_z [m]"]

            # Get displacement values for each compression step
            U_z[i-1] = self.get_U_z(data, L, W, H) # unit: m

            # Get negative poisson ratio constraint for last 
            npr[i-1] = self.get_npr(data, L, W, H) # unit: m

            # Get directional deformation
            dir_deform[i-1] = self.direction_deformation(data, L, W, H)
        
        # Make arrays to integrate Force-displacement courve
        y = np.concatenate((np.array([0]), -np.asarray(self.Loads)))
        x = np.concatenate((np.array([0]), -U_z),axis=0)
        Energy_absorbed = simpson(y=y, x=x) #(y,x) input for some reason, unit: N*m=J

        strain = x*1e3/H # unit: mm/mm = unitless (Compression strain is usually designated as negative [-])
        stress = y/(L*W) # unit: N/mm^2 = MPa
        gradiants = (stress[1:]-stress[:-1])/(strain[1:]-strain[:-1])
        E_c = np.mean(gradiants) # Estimated Compressive Elasticity modulus, unit: MPa

        return stress_vars, stress_maxs, Energy_absorbed, E_c, npr, dir_deform

    def reward_calc(self, s_vars, s_maxs, E_abs, E_c, npr, dir_deform, rel_density, action):
        # Calculates the reward for the RL training
        # To optimize training, this function/weights should be tuned as desired - significant time

        # To ensure comptabitabilty:
        action = np.array(action, dtype=np.float32)

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
        npr = npr[-1] # Negative Poisson Ratio criterion - Avearge displacement towards center from center of side walls
        dir = dir_deform[-1] # Directional deformation criterie - rightwards (U_x +) is positive
        # Negative components (The lower the better)
        Lw = action[2] * 1/3 # Length of lattice [mm] * weight
        Ww = action[3] * 1/3 # Width of lattice [mm] * weight
        Hw = action[4] * 1/3 # Height of lattice [mm] * weight
        sv = sv_algorithm(s_vars) * 1/20 # Variance of stress fields [MPa^2] * weight
        sm = sum(s_maxs > self.Elastic50a[3]) * 10 # When max stress [MPa] is above UTS * weight
        bt = action[0] * 15 # Beam thickness [mm] * weight
        cc = action[1] * 1/30 # Voronoi Cell Count [] * weight
        rd = rel_density * 5 # Relative Density [] * weight

        # Manually change and tailor this function to optimize for desired values
        # reward = Ec + Ea + np - Lw - Ww - Hw - sv - sm - bt - cc - rd - dm

        # reward = npr # unit [mm] - For negative poisson's ratio reward
        # reward = Ec # unit [MPa/50] - for optimal (high) stiffness
        # reward = dir # unit [mm] - rightwards movement / controlled deformation

        reward = #Define appropriate reward value

        # Protection against infinite reward
        if not np.isfinite(reward):
            print("Bad reward:", reward)
            reward = reward_calc_failed()

        return float(reward)

# Callback to track rewards during training. Necessary for RL to work properly
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

def reward_calc_failed():
    # Small function to give mean or -5 reward in case simulations failed due to meshing or infinite reward gradient.
    if len(reward_logger.episode_rewards) != 0:
        reward = np.mean(reward_logger.episode_rewards)
    else:
        reward = -5 # Chosen because roughly equal to worst possible reward froms succesfull simulation
    # Currently, fail always leads to reward of -5
    # reward = -5 # For reward=npr
    # reward = 0 # For reward=Ec & Control deformation
    reward = #Define appropriate reward for failed simulation. maybe just 0.
    return reward

#%% Train the RL model

# Plot optimal actions function:
def opt_design(savedmodel, i, episodes_per_figure):
    # Re-initialize environment
    env = LatticeEnv()
    state, _ = env.reset()

    # Load the saved model - used when want to predict using previously trained model
    RL_model = PPO.load(savedmodel, env=env, device="cpu")

    # Predict using RL_model
    action, _states = RL_model.predict(state, deterministic=True)
    action = env.denormalize_action(action)
    print(f"Optimal action for a {action[2]}mm x {action[3]}mm x {action[4]}mm (Length x Width x Height) design space is as such:")
    print(f"Beam Thickness:                     {action[0]:.4g}")
    print(f"Cell count:                         {action[1]:.0f}")

    # Save the Optimal density field as "ramp_input.csv" in the RL_training folder.
    LatticeEnv.generate_density_field(LatticeEnv, action)

    x = np.linspace(0,1,5)
    y = np.linspace(0,1,5)
    X, Y = np.meshgrid(x,y)
    Z = action[5:].reshape(5,5)
    contour = plt.contourf(X, Y, Z, cmap='jet', levels=100)
    plt.colorbar(contour, label="Relative Average Distance Between Seeds")
    plt.title(f'Episode {(i+1)*episodes_per_figure + 350}')
    plt.axis("equal")
    plt.tight_layout()
    plt.show()



# If one wants to get optimal actions out every X episode
episodes_per_figure = #50 # How many episodes per loop - Needs to be even number. 10 episodes ~ 0.8 hours
number_loops = #10 # How many loops
filename = #"new_ppo_lattice_model"
reward_tracer = np.zeros(episodes_per_figure*number_loops)
for i in range(number_loops):
    env = DummyVecEnv([lambda: LatticeEnv()])
    env = VecMonitor(env)

    t1 = time.time()
    #Defining the PPO agent (A2C - Advantage Actor Critic agent with Proximal Policy Optimization of a Multi-Layer Perceptron NN Policy)
    RL_model = PPO("MlpPolicy", env, verbose=1,
                    tensorboard_log="./ppo_lattice_tensorboard/",
                    batch_size=2, n_steps=2,
                    device="cpu")

    # To continue previous saved training.
    if os.path.exists(filename):
        RL_model = PPO.load(filename, env=env, device="cpu")

    RL_model.learn(total_timesteps=episodes_per_figure, callback=reward_logger) #total_timesteps is number of episodes
    t2 = time.time()
    print(f"Training time:\n{t2-t1:.5g} seconds or\n{(t2-t1)/60:.4g} minutes or\n{(t2-t1)/(60*60):.3g} hours")

    RL_model.save(filename)

    # Plot reward progression
    plt.plot(reward_logger.episode_rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Progression During Training")
    plt.legend()
    plt.show()

    opt_design(filename, i, episodes_per_figure)

# Save reward file manually if desired - specify new datafile name 
np.savetxt(#"new_0_to_500.csv", reward_logger.episode_rewards, delimiter=",")

# One can now run these through ntop to generate the optimal lattice structure