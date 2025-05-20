# Compare Simulations with Pre-thesis Compression data Batch 3:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


batch3_file = "Emil_lattices_18122024_SLA.xls"

xls_batch3 = pd.ExcelFile(batch3_file)


def extract_curves(xls_file):
    sheet_names = xls_file.sheet_names
    curve_data = {}
    
    for sheet in sheet_names:
        if sheet.lower() != "results":  # Skip overview sheet
            try:
                df = pd.read_excel(xls_file, sheet_name=sheet, skiprows=2)
                strain = df.iloc[:, 0] / 100  # Convert to decimal strain
                force = df.iloc[:, 1] # Convert to N
                
                stress = force / (25*25) #N/mm^2 = MPa

                # strain = strain[strain > 0.03] - 0.03
                stress = stress.loc[strain.index] - min(stress.loc[strain.index])

                sampled_strain = np.linspace(strain.min(), 0.15, 31)
                sampled_stress = np.interp(sampled_strain, strain, stress)

                 # Rename the curve based on batch
                if xls_file == xls_batch3:
                    new_name = f"b3{sheet.replace(' SLA', '')}"  # Remove ' SLA'
                
                curve_data[new_name] = (sampled_strain, sampled_stress)


            except Exception as e:
                print(f"Error reading {sheet}: {e}")

    return curve_data

curves_batch3 = extract_curves(xls_batch3)

def extract_e_compression(xls_file):
    df = pd.read_excel(xls_file, sheet_name="Results", usecols="B", skiprows=1)
    return df.squeeze().dropna().to_dict()

e_com_batch3 = extract_e_compression(xls_batch3) # MPa

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

#
v1 = ['b3v1r1','b3v1r2','b3v1r3']
v2 = ['b3v2r1','b3v2r2','b3v2r3']
v3 = ['b3v3r1','b3v3r2','b3v3r3']

#%% Gather Simulation data, and Plot comparison between sim and test
for sim_version,version in zip(['v1','v2','v3'],[v1,v2,v3]):
    L, W, H = 25, 25, 30
    Loads = (-np.array([15,30,45])).tolist()
    N_disp_sims = len(Loads) # Number of Specified Load simulations in nTop file
    U_z = np.zeros(N_disp_sims) # Initialize array for data
    npr = np.zeros(N_disp_sims) # Initialize array for data

    stress = np.zeros((3,N_disp_sims+1))
    strain = np.zeros((3,N_disp_sims+1))
    E_c = np.zeros(3)

    for j,rs in zip(range(3),['r1','r2','r3']):
        # Displacement Fields - Import from nTop and utilize:

        for i in range(1, 1+N_disp_sims):
            disp_file = "sim_results_batch3\\"+sim_version+rs+f"_displacement_{i}.csv"
            data = pd.read_csv(disp_file, header=None)
            data.columns = ["X [mm]", "Y [mm]", "Z [mm]", "U_x [m]", "U_y [m]", "U_z [m]"]

            # Get displacement values for each compression step
            U_z[i-1] = get_U_z(data, L, W, H)

        # Make arrays to integrate Force-displacement courve
        y = np.concatenate((np.array([0]), -np.asarray(Loads)))
        x = np.concatenate((np.array([0]), -U_z),axis=0)

        strain[j,:] = x*1e3/H # unitless (Compression strain is usually designated as negative [-])
        stress[j,:] = y/(L*W) # unit: N/mm^2 = MPa
        gradiants = (stress[j,1:]-stress[j,:-1])/(strain[j,1:]-strain[j,:-1])
        E_c[j] = np.mean(gradiants) #Estimated Compressive Elasticity modulus, unit: MPa

    plt.figure()
    colors = ['blue','red','green']
    for rs,j,c in zip(version,range(3),colors):
        plt.plot(curves_batch3[rs][0], curves_batch3[rs][1]*1e3, label='Test_'+rs, color=c)
        plt.plot(strain[j,:], stress[j,:]*1e3, linestyle='dotted',label=f'Simulation_b3{sim_version}r{j+1}',color=c)

    plt.legend()
    plt.xlabel('Compressive Strain')
    plt.ylabel('Applied Stress [kPa]')
    plt.xlim([0.00, 0.15])
    plt.grid()
    name = r'C:\Users\emilf\OneDrive - Aarhus universitet\Uni\10. Semester\Graphics\Comparisons_test_sim_b3'+sim_version
    plt.savefig(name+'.png')
    plt.show()
    
    #Comparing E_cs
    print('E_c comparisons for',sim_version)
    print(f'r1 Compression test: {e_com_batch3[(int(sim_version[-1])-1)*3+0]:.4g} MPa vs Simulation: {E_c[0]:.4g} MPa')
    print(f'r2 Compression test: {e_com_batch3[(int(sim_version[-1])-1)*3+1]:.4g} MPa vs Simulation: {E_c[1]:.4g} MPa')
    print(f'r3 Compression test: {e_com_batch3[(int(sim_version[-1])-1)*3+2]:.4g} MPa vs Simulation: {E_c[2]:.4g} MPa')