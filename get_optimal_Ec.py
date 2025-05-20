import numpy as np
import pandas as pd
import os
from scipy.integrate import simpson

RL_folder = "RL_training_folder\\" # Folder to keep all intermediate files for training (json, csv, stl, png)
This_folder = os.getcwd()
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  # nTopCL path
nTopFilePath = "lattice_auto_v3.ntop"  # Path to your nTop file

Compressions = (-np.array([0.025, 0.050, 0.075, 0.10, 0.125])).tolist() # Compressive strain
Loads = (-np.array([4, 8, 12])).tolist() # Newton - For Elastic50q material

def get_U_z(data, L, W, H):
    # Compute distances to center of load face. Will be at (0, W/2, H/2)
    data["distance_ct"] = np.sqrt((data["X [mm]"] - 0)**2 +
                                    (data["Y [mm]"] - W/2)**2 +
                                    (data["Z [mm]"] - H/2)**2)

        # Find the row with the minimum distance - to take compression at center of load face
    closest_row = data.loc[data["distance_ct"].idxmin()]

    # Get the corresponding U_z value
    U_z = closest_row["U_z [m]"]
    return U_z

def direction_deformation(data, L, W, H):
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


def ntop_sims(action):
    # Function to import and analyze the FEA simulation data from nTop.
    L, W, H = action[2], action[3], action[4] #Length Width and Height of current lattice structure

    # Stress Fields - Import from nTop and utilize:
    N_stress_sims = len(Compressions) # Number of Specified Compression simulations in nTop file
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
    N_disp_sims = len(Loads) # Number of Specified Load simulations in nTop file
    U_z = np.zeros(N_disp_sims) # Initialize array for data
    npr = np.zeros(N_disp_sims) # Initialize array for data
    dir_deform = np.zeros(N_disp_sims) # Initialize array for data
    for i in range(1, 1+N_disp_sims):
        disp_file = RL_folder+f"displacement_{i}.csv"
        data = pd.read_csv(disp_file, header=None)
        data.columns = ["X [mm]", "Y [mm]", "Z [mm]", "U_x [m]", "U_y [m]", "U_z [m]"]

        # Get displacement values for each compression step
        U_z[i-1] = get_U_z(data, L, W, H) # unit: m

        # Get directional deformation
        dir_deform[i-1] = direction_deformation(data, L, W, H)

        
    # Make arrays to integrate Force-displacement courve
    y = np.concatenate((np.array([0]), -np.asarray(Loads)))
    x = np.concatenate((np.array([0]), -U_z),axis=0)
    Energy_absorbed = simpson(y=y, x=x) #(y,x) input for some reason, unit: N*m=J

    strain = x*1e3/H # unit: mm/mm = unitless (Compression strain is usually designated as negative [-])
    stress = y/(L*W) # unit: N/mm^2 = MPa
    gradiants = (stress[1:]-stress[:-1])/(strain[1:]-strain[:-1])
    E_c = np.mean(gradiants) # Estimated Compressive Elasticity modulus, unit: MPa

    return stress_vars, stress_maxs, Energy_absorbed, E_c, npr, dir_deform


action_ = np.array([0.593, 129, 20, 5, 20])

stress_vars, stress_maxs, Energy_absorbed, E_c, npr, dir_deform = ntop_sims(action_)
