import os
import subprocess
import json
import pandas as pd

# Paths
csv_file_path = "lattice_parameters.csv"  # CSV input file
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  # nTopCL path
nTopFilePath = "lattice_auto.ntop"  # Path to your nTop file
json_folder = "json_files"  # Folder to store JSON files
summary_file = "output_summary.txt"  # Summary file
stl_output_folder = "stl_files" # Folder for new stl files

# Ensure json_files folder exists
os.makedirs(json_folder, exist_ok=True)
# Ensure stl_files folder exists, for ntop exports
os.makedirs(stl_output_folder, exist_ok=True)

# Load CSV
print("Reading CSV file...")
df = pd.read_csv(csv_file_path, delimiter=",") #Change to ; for param_input_test

# Open summary file
with open(summary_file, "w") as summary:
    
    # Iterate over each row in CSV
    for index, row in df.iterrows():
        input_filename = os.path.join(json_folder, f"input_{index+1}.json")
        output_filename = os.path.join(json_folder, f"output_{index+1}.json")
        
        # Construct input JSON
        input_json = {
            "inputs": [
                {"name": "Beam Thickness", "type": "scalar", "values": row["Beam Thickness"], "units": "mm"},
                {"name": "Voronoi Cell Count", "type": "integer", "values": row["Voronoi Cell Count"]},
                {"name": "Random Seed", "type": "integer", "values": row["Random Seed"]},
                {"name": "Name", "type": "text", "value": row["Name"]},
                {"name": "Length", "type": "scalar", "values": row["Length"], "units": "mm"},
                {"name": "Width", "type": "scalar", "values": row["Width"], "units": "mm"},
                {"name": "Height", "type": "scalar", "values": row["Height"], "units": "mm"},
                {"name": "Save Folder Path", "type": "text", "value": stl_output_folder}
            ]
        }
        
        # Save JSON file
        with open(input_filename, 'w') as f:
            json.dump(input_json, f, indent=4)
        
        # Construct nTopCL command
        command = [exePath, "-j", input_filename, "-o", output_filename, "-v1", nTopFilePath]
        
        # Run nTopCL
        print(f"Running nTopology for {row['Name']}...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Extract "val" from the output JSON
        try:
            with open(output_filename, 'r') as f:
                output_data = json.load(f)
                val = output_data[0]["value"]["val"]
                rounded_val = round(val, 3)  # Round to 3 significant digits
        except (KeyError, IndexError, json.JSONDecodeError):
            rounded_val = "ERROR"
        
        # Save extracted value in summary file
        summary.write(f"{row['Name']}: {rounded_val}\n")
        print(f"{row['Name']}: {rounded_val}")
    
print("Processing complete. Summary saved to", summary_file)
