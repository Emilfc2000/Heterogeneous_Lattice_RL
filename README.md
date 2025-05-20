# Heterogeneous Lattice RL
Reinforcement Learning Framework for designing optimized heterogeneous lattice structure utilizing Voronoi seeding theory. Requires a full access license for nTopology (Educational licences gives this)

This repository holds an RL design framework for heterogeneous lattice structures. Several versions are included, which have different features and capabilities:

Framework v1: Based on Reinforcement Learning and Convolutional Neural Network. The framework with the highest technical complexity and hardest to run. Requires large data to train a CNN to function as part of the RL environment. It should theoretically run fastest once running, but is currently only made to simply optimize compressive strength. Needs 1000's of data and tuning to work

Framework v2: Based on Reinforcement Learning and nTop compression FEA simulations. This is the simplest framework, and only requires a python code and an ntop file to run. Evaluation of the generated lattices is performed within the nTop file, and the result is returned to the RL Agent for rapid reward calculation. Reward calculation requires tuning.

Framework v3: Based on Reinforcement Learning, nTop compression FEA simulations, and controllable density field. This version is an upgrade to the v2 framework, and has the added functionality of defining a large number of control inputs to set the density field of the lattice structure, adding localized control parameters for the RL Agent to potentially control local behavior of the structure.

--------------------
**Framework Version 1: RL + CNN + data**

To Run Reinforcement Learning framework v1 (with CNN + data) one must:

1: Generate training data by creating lattice structures and compressing them (or simulations)

2: Utilize the stl_to_png Python file to generate PNG files for CNN to read

3: Utilize Extract_curves_and_E_c to extract the targets for the PNGs, such that CNN can be trained

4: Utilize flip_images_artificial_data to 4x amount of data, by mirroring all pngs in x y and z planes and copying the targets

5: Update all path and file locations in the scripts

6: Run Hybrid_CNN_NN to train the CNN

7: Download lattice_auto_v1.ntop

8: Set total_timesteps in Reinforcement_Learning_v1, and run it for the desired training duration

--------------------
**Framework Version 2: RL + nTop simulation**

To run Reinforcement Learning Framework v2: (with ntop simulations), one must:

1: Download lattice_auto_v2.ntop AND Reinforcement_Learning_v2.py, and put them in the same folder

2: Go into the Python file, specify the location of ntopCL on PC (at the beginning of the code)

3: Set total time_steps in the code (at the bottom) - this indicates the number of episodes the RL code will run. On a brand new high-end gaming laptop, one episode is roughly 500 seconds

4: Run the RL code, which will optimize the design parameters. To specify limits, one can also change the range of the design parameters at the beginning of the environment class.

--------------------
**Framework Version 3: RL + nTop simulation + controllable density field**

To run Reinforcement Learning Framework v3: (with ntop simulations AND controllable lattice density field), one must:

1: Download lattice_auto_v3.ntop AND Reinforcement_Learning_v3.py, and put them in the same folder

2: Go into the Python file, specify the location of ntopCL on PC (at the beginning of the code)

3: Set total time_steps in the code (at the bottom) - this indicates the number of episodes the RL code will run. On a brand new high-end gaming laptop, one episode is roughly 400 seconds

4: Run the RL code, and it will optimize the design parameters. One can also change the range of the design parameters in the begging of the environment class, to specify limits.
--------------------

## Python files:

Generate_csv_w_lattice_parameters.py - generate csv file with design parameters for desired number of lattice structures "lattice_parameters.csv"

Automate_Code_v2.py - uses lattice_parameters.csv to run all the design parameters through nTopology and generate a stl file describing lattice of each set of input parameters. Also generates "output_summary.txt" - a list of all generated lattices and their relative density. This also generated the json files in the json_files folder, which are used through ntop.

stl_to_png_v2.py - Takes all stl files in a specified folder, and generates Binary 512x512 png images from them

Extract_curves_and_E_c.py - generates processed_data.npz file with all 29 Force-strain curves and E_com values, using the data from the compressions tests

flip_images_artificial_data.py - generates artificial data to 4x the dataset for CNN. rotates/flips all lattices in x y and z direction, and copies the stress-strain curves. Creates the processed_data_expanded.npz with target data for CNN.

Hybrid_CNN_NN.py - Trains CNN and NN with PNG + specified material values as input, and data from npz as targets

Reinforcement_Learning_v1.py - The framework which utilizes the Hybrid CNN NN to train the RL Agent.

Reinforcement_Learning_v2.py - The framework which utilizes ntop simulations to train the RL Agent.

Reinforcement_Learning_v3.py - The framework which utilizes ntop simulations to train the RL Agent, which also has control of the density field of lattice cell distribution.

Voronoi_Lattice_2D.py - A separate, unrelated script, which, from input dimensions and parameters, can generate a 2D Voronoi lattice as a CAD drawing file. This could potentially be upgraded in the future to replace the nTop software for generating the lattices.

--------------------

Other Files:

lattice_auto_v1.ntop - nTop file setup to take in the JSON files generated by the RL agent and generate a lattice structure.

lattice_auto_v2.ntop - nTop file setup to take in the JSON files generated by the RL agent and generates a lattice structure, and performs a series of compression simulations

lattice_auto_v3.ntop - nTop file setup to take in the JSON files generated by the RL agent and generates a lattice structure with control of lattice density field, and performs a series of compression simulations.

--------------------

## Folders referenced in codes:

0_initial_docs_previous_version - Old Python documents and files no longer used.

All_pre_thesis_png - folder with PNGs of all 29 pre-thesis specimens.

All_pre_thesis_png_expanded - 29*4=116 pngs, including the original 29 pngs and the artificially generated data from flip_images_arti....py.

Batch_4_stl - STL files for all batches for lattice structures.

CNN_training_png - folder with all PNGs to be used for CNN training. 29 pre-thesis samples have been manually copied into this folder, and all future batches will also be put here.

CNN_training_png_expanded - the expanded version of the previous folder, with the artificial extra data.

json_files - Folder to hold the automatically generated json files which are used to automate the nTop design through the automade code.

png_images - currently holds a few lattices used to validate CNN results. 

RL_training_folder - folder which contains all intermediate files for RL training (CSV, JSON, STL, PNG).

Pre_thesis_data.zip - zip folder containing all data gathered in pre-thesis, which has been used ofr preliminary testing, development, and validation of simulations

prime_RL_heterogenous - folder to be copied into PRIME for training on cluster. Data for training should be placed in this folder under correct name and format.
