# Heterogeneous_Lattice_RL
Reinforcement Learning Framework for designed optimized heterogeneous lattice structure utilizing Voronoi seeding theory

Python files:

Generate_csv_w_lattice_parameters.py - generate csv file with design parameters for desired number of lattice structures "lattice_parameters.csv"

Automate_Code_v2.py - uses lattice_parameters.csv to run all the design parameters through nTopology and generate a stl file describing lattice of each set of input parameters. Also generates "output_summary.txt" - a list of all generated lattices and their relative density. This also generated the json files in the json_files folder, which are used through ntop.

stl_to_png_v2.py - Takes all stl files in a specified folder, and generates Binary 512x512 png images from them

Extract_curves_and_E_c.py - generates processed_data.npz file with all 29 Force-strain curves and E_com values, using the data from the compressions tests

flip_images_artificial_data.py - generates artificial data to 4x the dataset for CNN. rotates/flips all lattices in x y and z direction, and copies the stress-strain curves. Creates the processed_data_expanded.npz with target data for CNN.

Hybrid_CNN_NN.py - Trains CNN and NN with PNG + specified material values as input, and data from npz as targets

Reinforcement_Learning_code.py - The actual RL framework which utilizes the Hybrid CNN NN to train the RL Agent.



Folders:

0_initial_docs_previous_version - Old python documents and files no longer used.

All_pre_thesis_png - folder with pngs of all 29 pre-thesis specimens

All_pre_thesis_png_expanded - 29*4=116 pngs, including the original 29 pngs and the artificial generated data from flip_images_arti....py

Batch_4_stl - stl files for all batch for lattice structures

CNN_training_png - folder with all pngs to be used for CNN training. 29 pre-thesis samples have been manually copied into this folder, and all future batches will also be put here

CNN_training_png_expanded - the expanded version of the previous folder, with the artificial extra data

json_files - Folder to hold the automatically generated json files which are used to automate the nTop design through the automade code

png_images - currently holds a few lattices used to validate CNN results. 

RL_folder - folder which contains all intermediate files for RL training (json, stl, png)

__pycache__ - folder made by RL library used in training

ppo_lattice_tensorboard - folder made by RL library used in training

ppo_lattice_model - saved ppo model after RL training
