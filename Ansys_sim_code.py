import os
from ansys.mapdl.core import launch_mapdl

# Launch ANSYS MAPDL
mapdl = launch_mapdl(exec_file=r"C:\Program Files\ANSYS Inc\v241\ANSYS\bin\winx64\MAPDL.exe")

# Import STEP file
mesh_file = r"C:\Users\emilf\OneDrive - Aarhus universitet\Uni\10. Semester\Codes_and_files\RL_training_folder\test.igs"

# Check if file exists before trying to import
if not os.path.exists(mesh_file):
    raise FileNotFoundError(f"STEP file not found at {mesh_file}")

# Switch to pre-processing
# mapdl.prep7()

# Import STEP file
mapdl.aux15()
mapdl.igesin(mesh_file)
# mapdl.aux15()
mapdl.prep7()  # Exit AUX15 and return to PREP7

# Define Material Properties (BlackV4 example)
mapdl.mp("EX", 1, 2.6e9)   # Young's Modulus (Pa)
mapdl.mp("PRXY", 1, 0.35)   # Poisson's Ratio
mapdl.mp("DENS", 1, 1200)  # Density (kg/m³)

# Define Tensile Ultimate Strength
# mapdl.tb("PLASTIC", 1, 1, "", "MISO")  # Define plasticity table
# mapdl.tbdata(1, 65e6, 0.0)  # Tensile Ultimate Strength (Pa)

# Define Geometry and Mesh
mapdl.et(1, 185)  # Define element type (SOLID185)
mapdl.esize(0.5)   # Set mesh element size
mapdl.vmesh("ALL")  # Mesh the volume


# Apply Boundary Conditions
mapdl.nsel("S", "LOC", "Z", 0)  # Select nodes at Z = 0 (bottom)
mapdl.d("ALL", "ALL")  # Apply fixed support

# Apply Force on Top Face
H = 27  # Height of Sample (Should be sampled from RL generation)
mapdl.nsel("S", "LOC", "Z", H)  # Select nodes at top
mapdl.d("ALL", "UZ", -0.15 * H)  # Apply 15% compression (negative Z)

# Solve the Simulation
mapdl.allsel()
mapdl.solve()
mapdl.finish()

# Post-Processing: Get Displacement Results
mapdl.post1()
mapdl.set(1, 1)
disp = mapdl.get("Umax", "NODE", 0, "U", "SUM")  # Get max displacement
print(f"Maximum Displacement: {disp} meters")

# Exit ANSYS
mapdl.exit()
