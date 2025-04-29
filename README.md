# MTPrompt-PTM

This is the official implementation of MTPrompt-PTM. MTPrompt-PTM is a multi-task PTM prediction framework developed by applying prompt tuning to a Structure-Aware Protein Language Model (S-PLM). Instead of training several single-task models, MTPrompt-PTM trains one multi-task model to predict multiple types of PTM site using shared feature-extraction layers and task-specific classification heads. 

The PTM sites includes Phosphorylation (S, T, Y), N-linked Glycosylation (N), O-linked Glycosylation (S, T), Ubiquitination (K), Acetylation (K), Methylation (K, R), SUMOylation (K), Succinylation (K), and Palmitoylation (C).

<h2>Architecture</h2>
<img width="800" alt="image" src="https://github.com/hanye311/MTPrompt-PTM/blob/main/Architecture.jpg" />

<h2>Installation</h2>

To use MTPrompt-PTM project, install the corresponding <code>environment.yaml</code> file in your environment. Or you can follow the <code>install.sh</code> file to install the dependencies.

<h3>Install using yaml file</h3>

Using <code>environment.yaml</code>

1. Create a new environment using the environment.yaml file: <code>conda env create -f environment.yaml</code>.
2. Activate the environment you have just created: <code>conda activate mtprompt</code>.

<h3>Install using SH file</h3>

Create a conda environment and use this command to install the required packages inside the conda environment. First, make the <code>install.sh</code> file executable by running the following command:

```bash
chmod +x install.sh
```
Then, run the following command to install the required packages inside the conda environment:

```bash
bash install.sh
```

<h2>Inference</h2>

To predict PTM sites using our model, you need to provide the PTM type and the corresponding protein sequence.

The available PTM types are as follows: 
Phosphorylation_S, Phosphorylation_T, Phosphorylation_Y, Ubiquitination_K, Acetylation_K, OlinkedGlycosylation_S, Methylation_R, NlinkedGlycosylation_N, OlinkedGlycosylation_T, Methylation_K, Palmitoylation_C, Sumoylation_K and Succinylation_K.

The protein sequence must be saved in a FASTA file format.

The model can be downloaded from: [MTPrompt-PTM](https://mailmissouri-my.sharepoint.com/:u:/g/personal/yhhdb_umsystem_edu/EeDvZmGms2dJkOW_ob_0MOcBXvFhm5evkAYgz3shfNAheA?e=vtMAJg).

The result will be saved in a csv file.

```bash
python test.py --config_path config/PTM_config_prompt_tuning_test.yaml --model_path best_model_13ptm_final.pth --data_path data/Phosphorylation_S_sequence.fasta --PTM_type Phosphorylation_S --save_path data
```
