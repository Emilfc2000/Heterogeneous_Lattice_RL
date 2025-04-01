# Heterogeneous Lattice RL
Reinforcement Learning Framework for designed optimized heterogeneous lattice structure utilizing Voronoi seeding theory. Requires full access license for nTopology.

This repository currently holds 2 different RL frameworks. One based fully on Python RL and ntop (lattice generation and simulation), and another which is based on PYthon RL and CNN, ntop (lattice generation), and is missing some data / simulation software (could be ANSYS), to generate data for the CNN. If the CNN is properly trained, this framework will run faster. However, the first (only python+ntop) also runs well, and works currently - though pending a tuning of reward-function.

--------------------

To Run Reinforcement Learning with CNN + data framework, one must:

1: Generate training data by creating lattice structures and compression testing them (or simulations)

2: Utilize stl_to_png python file to generate png files for CNN to read

3: Utilize Extract_curves_and_E_c to extract the targets for the pngs, such that CNN can be trained

4: Utilize flip_images_artificial_data to 4x amount of data, by mirroring all pngs in x y and z planes and copying the targets

5: Update all path and file locations in the scripts

6: Run Hybrid_CNN_NN to train CNN

7: set total_timesteps in Reinforcement_Learning_code, and run it for the desired training duration

--------------------

To Run Reinforcement Learning with ntop simulations framework, one must:

1: Download lattice_auto_w_simulations.ntop AND Reinforcement_Learning_code_ntop_sim.py, and put them in the same folder

2: Go into python file, specify location of ntopCL on PC (in the begging of the code)

3: Set total time_steps in the code (at the bottom) - this indicates the number of episodes the RL code will run. On brand new high end gaming laptop, one episode i roughly 500 seconds

4: Run the RL code, and it will optimize the design parameters. One can also change the range of the design parameters in the begging of the environment class, to specify limits.

--------------------

To run on PRIME cluster - DOES NOT WORK CURRENTLY

Go to prime_RL_heterogneous folder - ensure RL_prime_code has correct total_timesteps and other parameters.

Ensure all training data (pngs and npz data files with E_c and curves) are in correct format and has correct names to interact with RL and CNN code

scp the prime_RL_heterogeneous folder to prime /com/meadd (This folder includes slightly varied versions of the necessary files for PRIME to run: RL code, CNN code, nTop file, training pngs and target curves and E_cs.)

(Is done by writing on terminal on own PC: "scp my_pc_path/to_file.txt auXXXXXX@prime.eng.au.dk:path_to/location *prime_password*)

Simply run the RL_prime_code

--------------------
Accessing PRIME cluster

Open Terminal

"ssh auXXXXXX@ssh.au.dk"

"*au-password*"

Accept 2-Factor-Authentication

"ssh auXXXXXX@prime.eng.au.dk"

"*Prime-Password*"

--------------------

## Python files:

Generate_csv_w_lattice_parameters.py - generate csv file with design parameters for desired number of lattice structures "lattice_parameters.csv"

Automate_Code_v2.py - uses lattice_parameters.csv to run all the design parameters through nTopology and generate a stl file describing lattice of each set of input parameters. Also generates "output_summary.txt" - a list of all generated lattices and their relative density. This also generated the json files in the json_files folder, which are used through ntop.

stl_to_png_v2.py - Takes all stl files in a specified folder, and generates Binary 512x512 png images from them

Extract_curves_and_E_c.py - generates processed_data.npz file with all 29 Force-strain curves and E_com values, using the data from the compressions tests

flip_images_artificial_data.py - generates artificial data to 4x the dataset for CNN. rotates/flips all lattices in x y and z direction, and copies the stress-strain curves. Creates the processed_data_expanded.npz with target data for CNN.

Hybrid_CNN_NN.py - Trains CNN and NN with PNG + specified material values as input, and data from npz as targets

Reinforcement_Learning_code_cnn.py - The 2nd RL framework which utilizes the Hybrid CNN NN to train the RL Agent.

Reinforcement_Leraning_code_ntop_sim.py - The 1st framework which utilizes ntop simulations to train the RL agent

Voronoi_Lattice_2D.py - A seperate unrelated script, which from input dimensions and parameters can generate a 2D voronoi lattice as a cad drawing file. This could potentially in the the future be upgraded to replace the nTop software for generating the lattices.

--------------------

Other Files:

lattice_auto.ntop - nTop file setup to take in the json files generated by the RL agent and generate lattice structure

lattice_auto.ntop - nTop file steup to take in the json files generated by the RL agent and generates lattice structure and performs a series of compression simulations

--------------------

## Folders:

0_initial_docs_previous_version - Old python documents and files no longer used.

All_pre_thesis_png - folder with pngs of all 29 pre-thesis specimens

All_pre_thesis_png_expanded - 29*4=116 pngs, including the original 29 pngs and the artificial generated data from flip_images_arti....py

Batch_4_stl - stl files for all batch for lattice structures

CNN_training_png - folder with all pngs to be used for CNN training. 29 pre-thesis samples have been manually copied into this folder, and all future batches will also be put here

CNN_training_png_expanded - the expanded version of the previous folder, with the artificial extra data

json_files - Folder to hold the automatically generated json files which are used to automate the nTop design through the automade code

png_images - currently holds a few lattices used to validate CNN results. 

RL_folder - folder which contains all intermediate files for RL training (json, stl, png)

prime_RL_heterogenous - folder to be copied into PRIME for training on cluster. Data for training should be placed in this folder under correct name and format.
