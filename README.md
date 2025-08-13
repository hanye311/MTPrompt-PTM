# MTPrompt-PTM

This is the official implementation of MTPrompt-PTM. MTPrompt-PTM is a multi-task PTM prediction framework developed by applying prompt tuning to a Structure-Aware Protein Language Model (S-PLM). Instead of training several single-task models, MTPrompt-PTM trains one multi-task model to predict multiple types of PTM site using shared feature-extraction layers and task-specific classification heads. 

The PTM sites includes Phosphorylation (S, T, Y), N-linked Glycosylation (N), O-linked Glycosylation (S, T), Ubiquitination (K), Acetylation (K), Methylation (K, R), SUMOylation (K), Succinylation (K), and Palmitoylation (C).

<h2>Architecture</h2>
<img width="800" alt="image" src="https://github.com/hanye311/MTPrompt-PTM/blob/main/Architecture.jpg" />

<h2>Installation</h2>

To use MTPrompt-PTM project, install the corresponding <code>environment.yml</code> file in your environment. Or you can follow the <code>requirements.txt</code> file to install the dependencies.

<h3>Install using yml file</h3>

Using <code>environment.yml</code>

1. Create a new environment using the environment.yml file: <code>conda env create -f environment.yml</code>.
2. Activate the environment you have just created: <code>conda activate mtprompt</code>.

<h3>Install using requirement file</h3>

First create (and activate) a clean virtual environment, then point pip at your requirements.txt

```bash
python3 -m venv venv
pip install -r requirements.txt
```

<h2>Inference</h2>

To predict PTM sites using our model, you need to provide the PTM type and the corresponding protein sequence.

The available PTM types are as follows: 
Phosphorylation_S, Phosphorylation_T, Phosphorylation_Y, Ubiquitination_K, Acetylation_K, OlinkedGlycosylation_S, Methylation_R, NlinkedGlycosylation_N, OlinkedGlycosylation_T, Methylation_K, Palmitoylation_C, Sumoylation_K and Succinylation_K.

The protein sequence must be saved in a FASTA file format.

The model can be downloaded from: [MTPrompt-PTM](https://drive.google.com/file/d/1FfMepaY1JLUbKTZncE1u7-pm2d16IUuf/view?usp=drive_link).

The result will be saved in a csv file.

```bash
python test.py --config_path config/PTM_config_prompt_tuning_test.yaml --model_path best_model_13ptm_final.pth --data_path data/Phosphorylation_S_sequence.fasta --PTM_type Phosphorylation_S --save_path data
```


<h3>Using Docker</h3>

```bash
docker run --rm --name mtprompt -v "$(pwd)":/app/data hanye0311/mtprompt:v1 python3 test.py --data_path /app/data/Phosphorylation_S_sequence.fasta --PTM_type Phosphorylation_S
```

<h3>Dataset</h3>

The dataset for training, validation and testing can be downloaded from: [Dataset](https://drive.google.com/drive/folders/14Cw81Fua7Gcb76dkrfxySF9hldPSZRqx?usp=drive_link).
