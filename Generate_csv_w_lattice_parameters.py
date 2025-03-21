import csv
import random
import numpy as np

# Define adjustable parameter ranges
num_samples = 5  # Number of unique lattice designs (each has 3 random seeds)
beam_thickness_range = (0.4, 0.6)
vcc_range = (70, 160)  # Higher VCC for lower BT, higher L/H
Number_of_Random_seeds = 1
length_values = np.linspace(20,30,11,dtype=int)
width_values = np.linspace(5,30,26,dtype=int)
height_values = np.linspace(20,30,11,dtype=int)

# Generate CSV file
def generate_csv(filename="lattice_parameters.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Beam Thickness", "Voronoi Cell Count", "Random Seed", "Name", "Length", "Width", "Height"])

        sample_id = 1  # Naming scheme base
        for i in range(num_samples):
            bt = round(random.uniform(*beam_thickness_range), 2)
            length = 20 #random.choice(length_values)
            width = 20 #random.choice(width_values)
            height = 20 #random.choice(height_values)
            
            # VCC adjustment based on BT
            vcc = int(vcc_range[1] - ((bt - beam_thickness_range[0]) / (beam_thickness_range[1] - beam_thickness_range[0])) * (vcc_range[1] - vcc_range[0]))
            vcc = max(vcc, vcc_range[0])  # Ensure within range
            
            for rs in range(1, Number_of_Random_seeds+1): # Generate x number of each design
                name = f"{sample_id}-{rs}"
                writer.writerow([bt, vcc, rs, name, length, width, height])
            
            sample_id += 1  # Increment sample ID

# Run the generator
generate_csv()
print("CSV file generated successfully!")
