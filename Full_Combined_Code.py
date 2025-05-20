# FULL CODE:
import subprocess

# Generate a csv file from the chosen input parameter code:
print("Generating csv file...")
subprocess.run(["python", "Generate_csv_w_lattice_parameters.py"])
print("Done - csv created")

# Run all these parameters through nTopCL to gather stl files and relative densities data
print("Running parameters through nTop...")
subprocess.run(["python", "Batch_4_Automate_Code_v2.py"])
print("Done - stl files generated & output_summary.txt updated")

# Generate png images of each lattice
print("Generate png images...")
subprocess.run(["python", "stl_to_png_v2.py"])
print("Done - pngs generated")