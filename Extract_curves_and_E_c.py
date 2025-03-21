import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# File paths for Excel data
batch2_file = r"C:\\Users\\emilf\\OneDrive - Aarhus universitet\\Uni\\9. Semester\\R&D Pre-Thesis\\Batch 2 Lattice Print Samples\\Test 1 Compression of Batch 2 - emil_lattices_04-11-2024.xls"
batch3_file = r"C:\\Users\\emilf\\OneDrive - Aarhus universitet\\Uni\\9. Semester\\R&D Pre-Thesis\\Batch 3 Lattice Print Samples\\Emil_lattices_18122024_SLA.xls"
# batch4_file = r"C:\\Users\\emilf\\OneDrive - Aarhus universitet\\Uni\\10. Semester\\....xls"

# Load Excel files
xls_batch2 = pd.ExcelFile(batch2_file)
xls_batch3 = pd.ExcelFile(batch3_file)
# xls_batch4 = pd.ExcelFile(batch4_file)

# Function to extract force-strain curves from a given Excel file
def extract_curves(xls_file):
    sheet_names = xls_file.sheet_names
    curve_data = {}
    
    for sheet in sheet_names:
        if sheet.lower() != "results":  # Skip overview sheet
            try:
                df = pd.read_excel(xls_file, sheet_name=sheet, skiprows=2)
                strain = df.iloc[:, 0] / 100  # Convert to decimal strain
                force = df.iloc[:, 1] # Convert to N
                if xls_file == xls_batch2:
                    stress = force / (20*20) #N/mm^2 = MPa
                elif xls_file == xls_batch3:
                    stress = force / (25*25) #N/mm^2 = MPa
                # elif xls_file == xls_batch4:
                #     if sheet.endswith("v1r1") or sheet.endswith("v1r2"):
                #         Area = (22*22) #mm*2
                #     if sheet.endswith("v2r1") or sheet.endswith("v2r2"):
                #         Area = (24*24) #mm*2
                #     if sheet.endswith("v3r1") or sheet.endswith("v3r2"):
                #         Area = (22*22) #mm*2
                #     if sheet.endswith("v4r1") or sheet.endswith("v4r2"):
                #         Area = (24*24) #mm*2
                #     if sheet.endswith("v5r1") or sheet.endswith("v5r2"):
                #         Area = (26*26) #mm*2
                #     if sheet.endswith("v6r1") or sheet.endswith("v6r2"):
                #         Area = (28*28) #mm*2
                #     if sheet.endswith("v7r1") or sheet.endswith("v7r2"):
                #         Area = (28*28) #mm*2
                #     if sheet.endswith("v8r1") or sheet.endswith("v8r2"):
                #         Area = (30*30) #mm*2
                #     stress = force / Area # N/mm^2 = MPa
                
                # Fix potential errors in strain data
                strain = strain[strain > 0.03] - 0.03
                stress = stress.loc[strain.index]
                
                # Sample 100 points from the curve
                sampled_strain = np.linspace(strain.min(), 0.3, 32) #only to 30% compression
                sampled_stress = np.interp(sampled_strain, strain, stress)

                 # Rename the curve based on batch
                if xls_file == xls_batch2:
                    new_name = f"b2{sheet}"
                elif xls_file == xls_batch3:
                    new_name = f"b3{sheet.replace(' SLA', '')}"  # Remove ' SLA'
                
                curve_data[new_name] = (sampled_strain, sampled_stress)
            except Exception as e:
                print(f"Error reading {sheet}: {e}")
    
    return curve_data

# Extract force-strain curves for both batches
curves_batch2 = extract_curves(xls_batch2)
curves_batch3 = extract_curves(xls_batch3)
# curves_batch4 = extract_curves(xls_batch4)

# Function to extract E_compression values from "Results" sheet
def extract_e_compression(xls_file):
    df = pd.read_excel(xls_file, sheet_name="Results", usecols="B", skiprows=1)
    return df.squeeze().dropna().to_dict()

# Extract E_compression values
e_com_batch2 = extract_e_compression(xls_batch2)
e_com_batch3 = extract_e_compression(xls_batch3)
# e_com_batch4 = extract_e_compression(xls_batch4)

# Save extracted data (with batch 4)
# np.savez("processed_data.npz", curves_batch2=curves_batch2, curves_batch3=curves_batch3, curves_batch4 = curves_batch4, 
#          e_com_batch2=e_com_batch2, e_com_batch3=e_com_batch3, e_com_batch4=e_com_batch4)

# Save extracted data (No batch 4)
np.savez("processed_data.npz", curves_batch2=curves_batch2, curves_batch3=curves_batch3, 
         e_com_batch2=e_com_batch2, e_com_batch3=e_com_batch3)

print("Data extraction complete!")

# Load saved data and plot some curves for verification
data = np.load("processed_data.npz", allow_pickle=True)
curves_batch2 = data["curves_batch2"].item()
curves_batch3 = data["curves_batch3"].item()
# curves_batch4 = data["curves_batch4"].item()
e_com_batch2 = data["e_com_batch2"].item()
e_com_batch3 = data["e_com_batch3"].item()
# e_com_batch4 = data["e_com_batch4"].item()


# Plot some random force-strain curves
plt.figure(figsize=(10, 5))
for i, (name, (strain, force)) in enumerate(curves_batch2.items()):
    if i >= 20:
        break
    plt.plot(strain, force, label=name)
plt.xlabel("Compression Strain [-]")
plt.ylabel("Stress Stress [MPa]")
plt.title("Sample Stress-Strain Curves (Batch 2)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
for i, (name, (strain, force)) in enumerate(curves_batch3.items()):
    if i >= 9:
        break
    plt.plot(strain, force, label=name)
plt.xlabel("Compression Strain [-]")
plt.ylabel("Stress Applied [MPa]")
plt.title("Sample Stress-Strain Curves (Batch 3)")
plt.legend()
plt.grid()
plt.show()

# plt.figure(figsize=(10, 5))
# for i, (name, (strain, force)) in enumerate(curves_batch4.items()):
#     if i >= 16:
#         break
#     plt.plot(strain, force, label=name)
# plt.xlabel("Compression Strain [-]")
# plt.ylabel("Stress Applied [MPa]")
# plt.title("Sample Stress-Strain Curves (Batch 3)")
# plt.legend()
# plt.grid()
# plt.show()

print(e_com_batch2)
print(e_com_batch3)
# print(e_com_batch4)

print("Data extraction and verification complete!")