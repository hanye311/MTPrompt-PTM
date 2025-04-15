import torch
import yaml
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_configs, calculate_class_weights
from transformers import AutoTokenizer, T5Tokenizer
from utils import truncate_seq
import os
import tqdm
import re
import esm
from esm import pretrained
from model import prepare_adapter_h_model
import pickle
from Bio import SeqIO
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def find_indexes(string, chars):
    """
    Returns the 1-indexed positions of specified characters in a string, excluding certain positions.

    Parameters:
    - string (str): The input string in which to search for characters.
    - chars (list of str): A list of characters to search for in the string.
    - exclude_indexes (list of int): A list of 1-indexed positions to exclude from the search.

    Returns:
    - list of int: A list of 1-indexed positions where the characters from 'chars' appear in 'string',
                   excluding positions from 'exclude_indexes'.

    """
    # Convert the 1-indexed positions to 0-indexed positions for Python

    indexes = []
    for i, char in enumerate(string):
        # Check if the character is in the chars list and not in the exclude list
        if char in chars:
            # Convert the 0-indexed position back to 1-indexed for the result
            indexes.append(i + 1)
    return indexes


def check_ptm_site(sequence, positions, allowed_ptm_sites):
    for i in positions.copy():
        if i > len(sequence):
            positions.remove(i)
            continue
        elif not sequence[i - 1] in allowed_ptm_sites:
            positions.remove(i)
    return positions

def check_center_amino_acid(sequence, position, positive_amino_acids):
   if sequence[position - 1] in positive_amino_acids:
       return True

def extract_positions(sequence,configs):
    ptm_position= {}
    token_info = configs.encoder.condition_token.token_info
    for i in range(len(sequence)):
        if sequence[i] in token_info.keys():
            ptm_position[i]=token_info[sequence[i]]
        else:
            ptm_position[i]="None"

    return ptm_position


class PTMDataset(Dataset):
    def __init__(self, samples_list, configs):
        self.samples_list=samples_list
        self.configs = configs

        if  self.configs.encoder.model_name.startswith("esmc"):
            self.max_length = configs.encoder.max_len

            # self.client = ESMC.from_pretrained("esmc_600m").to(device)  # or "cpu"
        else:
            # self.esm2_model, self.esm2_alphabet = prepare_adapter_h_model(configs_all, logging)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
            self.max_length = configs.encoder.max_len


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        prot_id, sequence, mask, task_token, index = (
        self.samples_list[index][0], self.samples_list[index][1], self.samples_list[index][2],
        self.samples_list[index][3],index)
        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )

        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        padded_mask = np.pad(mask, (0, self.max_length - len(sequence)), 'constant')
        return prot_id, encoded_sequence,  padded_mask.astype(bool), task_token,index

def prepare_task(dataset_path, task_token, positive_amino_acids):

    # Path to the FASTA file
    fasta_file = dataset_path
    data_list=[]

    # Reading the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        data_dic={}
        data_dic['id']=record.id
        data_dic['sequence']=str(record.seq)
        data_list.append(data_dic)
    df = pd.DataFrame(data_list)

    samples = []

    for row in df.itertuples():
        sequence = row.sequence
        prot_id = row.id
        valid_mask = [0] * len(sequence)

        all_positions = find_indexes(sequence, positive_amino_acids)
        for position in all_positions:
            # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
            valid_mask[position - 1] = 1

        samples.append((prot_id, sequence,  valid_mask, task_token))
    return samples

def get_test_samples(configs, task_info):
    test_samples = prepare_task(
        dataset_path=configs.data_path,
        task_token=task_info['id'],
        positive_amino_acids=task_info['ptm_amino_acid']
    )
    dataset_test = PTMDataset(test_samples, configs)
    test_loader = DataLoader(dataset_test, batch_size=configs.test_settings.batch_size,
                             shuffle=True, pin_memory=False, drop_last=False,
                             num_workers=configs.test_settings.num_workers)

    return test_loader

def prepare_dataloaders_ptm(args,configs):

    task_list=[]
    dataloaders_dict_test= {}

    if args.PTM_type=='Phosphorylation_S':
        # configs.tasks.Phosphorylation_S = True
        task_list.append(
            {'task_name':"Phosphorylation_S",'id':configs.task_ids.Phosphorylation_S,'file_name':args.data_path,
                          'ptm_amino_acid':["S"]})

    if args.PTM_type=="Phosphorylation_T":
        # configs.tasks.Phosphorylation_T = True
        task_list.append(
            {'task_name': "Phosphorylation_T", 'id': configs.task_ids.Phosphorylation_T, 'file_name': args.data_path,
             'ptm_amino_acid': ["T"]})

    if args.PTM_type=="Phosphorylation_Y":
        # configs.tasks.Phosphorylation_Y = True
        task_list.append(
            {'task_name': "Phosphorylation_Y", 'id': configs.task_ids.Phosphorylation_Y, 'file_name': args.data_path,
             'ptm_amino_acid': ["Y"]})

    if args.PTM_type=="Ubiquitination_K":
        # configs.tasks.Ubiquitination_K = True
        task_list.append(
            {'task_name': "Ubiquitination_K", 'id': configs.task_ids.Ubiquitination_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Acetylation_K":
        # configs.tasks.Acetylation_K = True
        task_list.append(
            {'task_name': "Acetylation_K", 'id': configs.task_ids.Acetylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="OlinkedGlycosylation_S":
        # configs.tasks.OlinkedGlycosylation_S = True
        task_list.append(
            {'task_name': "OlinkedGlycosylation_S", 'id': configs.task_ids.OlinkedGlycosylation_S, 'file_name': args.data_path,
             'ptm_amino_acid': ["S"]})

    if args.PTM_type=="Methylation_R":
        # configs.tasks.Methylation_R = True
        task_list.append(
            {'task_name': "Methylation_R", 'id': configs.task_ids.Methylation_R, 'file_name': args.data_path,
             'ptm_amino_acid': ["R"]})

    if args.PTM_type=="NlinkedGlycosylation_N":
        # configs.tasks.NlinkedGlycosylation_N = True
        task_list.append(
            {'task_name': "NlinkedGlycosylation_N", 'id': configs.task_ids.NlinkedGlycosylation_N, 'file_name': args.data_path,
             'ptm_amino_acid': ["N"]})

    if args.PTM_type=="OlinkedGlycosylation_T":
        # configs.tasks.OlinkedGlycosylation_T= True
        task_list.append(
            {'task_name': "OlinkedGlycosylation_T", 'id': configs.task_ids.OlinkedGlycosylation_T, 'file_name': args.data_path,
             'ptm_amino_acid': ["T"]})

    if args.PTM_type=="Methylation_K":
        # configs.tasks.Methylation_K = True
        task_list.append(
            {'task_name': "Methylation_K", 'id': configs.task_ids.Methylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Palmitoylation_C":
        # configs.tasks.Palmitoylation_C = True
        task_list.append(
            {'task_name': "Palmitoylation_C", 'id': configs.task_ids.Palmitoylation_C, 'file_name': args.data_path,
             'ptm_amino_acid': ["C"]})

    if args.PTM_type=="Sumoylation_K":
        # configs.tasks.Sumoylation_K = True
        task_list.append(
            {'task_name': "Sumoylation_K", 'id': configs.task_ids.Sumoylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    if args.PTM_type=="Succinylation_K":
        # configs.tasks.Succinylation_K = True
        task_list.append(
            {'task_name': "Succinylation_K", 'id': configs.task_ids.Succinylation_K, 'file_name': args.data_path,
             'ptm_amino_acid': ["K"]})

    for task_info in task_list:
        dataloaders_dict_test[task_info['task_name']] = get_test_samples(configs, task_info)

    return {'test': dataloaders_dict_test}

# if __name__ == '__main__':
#     config_path = './config_enzyme_reaction.yaml'
#     with open(config_path) as file:
#         configs_dict = yaml.full_load(file)
#
#     configs_file = load_configs(configs_dict)
#
#     dataloaders_dict = prepare_dataloaders_enzyme_reaction(configs_file)
#     max_position_value = []
#     amino_acid = []
#     for batch in dataloaders_dict['train']:
#         sequence_batch, label_batch, position_batch, weights_batch = batch
#         # print(sequence_batch['input_ids'].shape)
#         # print(label_batch.shape)
#         # print(position_batch.shape)
#         max_position_value.append(position_batch.squeeze().numpy().item())
#         amino_acid.append(sequence_batch["input_ids"][0][position_batch.squeeze().numpy().item()].item())
#     print(set(max_position_value))
#     print([dataloaders_dict['train'].dataset.encoder_tokenizer.id_to_token(i) for i in set(amino_acid)])
#     print('done')
