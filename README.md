# MTPrompt-PTM

This is the official implementation of MTPrompt-PTM. MTPrompt-PTM is a multi-task PTM prediction framework developed by applying prompt tuning to a Structure-Aware Protein Language Model (S-PLM). Instead of training several single-task models, MTPrompt-PTM trains one multi-task model to predict multiple types of PTM site using shared feature-extraction layers and task-specific classification heads. 

The PTM sites includes Phosphorylation (S, T, Y), N-linked Glycosylation (N), O-linked Glycosylation (S, T), Ubiquitination (K), Acetylation (K), Methylation (K, R), SUMOylation (K), Succinylation (K), and Palmitoylation (C).

<h2>Architecture</h2>
<img width="500" alt="image" src="https://github.com/hanye311/MTPrompt-PTM/blob/main/Architecture.jpg" />

<h2>Installation</h2>

To use S-PLM project, install the corresponding environment.yaml file in your environment. Or you can follow the install.sh file to install the dependencies.

<h3>Install using yaml file</h3>

Using environment.yaml

Create a new environment using the environment.yaml file: conda env create -f environment.yaml.
Activate the environment you have just created: conda activate splm.

<h3>Install using SH file</h3>

Create a conda environment and use this command to install the required packages inside the conda environment. First, make the install.sh file executable by running the following command:

chmod +x install.sh

Then, run the following command to install the required packages inside the conda environment:

bash install.sh

<h2>Run</h2>
