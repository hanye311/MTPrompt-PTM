import os
import numpy as np
import yaml
import argparse
import torch
from time import time
from tqdm import tqdm
from data import prepare_dataloaders_ptm
from model import prepare_models_secondary_structure_ptm
from torch.nn import functional as F
import pandas as pd
from box import Box
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict(dataloader, net):
    counter = 0
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description("Steps")

    prediction_results=[]
    prot_id_results=[]
    position_results=[]

    for i, data in enumerate(dataloader):
        prot_id,sequences, masks, task_ids,indices = data
        with torch.inference_mode():
            outputs = net(sequences, task_ids)
            batch_size =task_ids.size(0)
            final_preds=[]
            for i in range(batch_size):
                example_task_id = task_ids[i].item()
                logits = outputs[example_task_id][i][masks[i]]
                final_preds.append(logits)
                indices = (masks[i] == 1).nonzero(as_tuple=True)[0].tolist()
                positions=[v+1 for v in indices]
                position_results.extend(positions)

            preds = torch.cat(final_preds,dim=0)
            preds = F.softmax(preds, dim=-1)[:, 1]
            rounded_preds = [round(v, 3) for v in preds.tolist()]
            prediction_results.extend(rounded_preds)
            numbers_of_sequences=[final_preds[i].shape[0] for i in range(len(final_preds))]
            k = 0
            for prot in list(prot_id):
                for i in range(numbers_of_sequences[k]) :
                    prot_id_results.append(prot)
                k+=1
        counter += 1
        progress_bar.update(1)

    return prediction_results,prot_id_results,position_results


def main(args, dict_config):
    configs = Box(dict_config)

    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    dataloaders_dict = prepare_dataloaders_ptm(args,configs)
    net = prepare_models_secondary_structure_ptm(configs)
    model_checkpoint = torch.load(args.model_path, map_location='cpu')

    net.load_state_dict(model_checkpoint['model_state_dict'])

    for i, (task_name, dataloader) in enumerate(dataloaders_dict['test'].items()):
        net.eval()
        start_time = time()
        prediction_results,  prot_id_results,position_results = predict(dataloader, net)
        result_dic = {
            "prot_id": prot_id_results,
            "position":position_results,
            "prediction": prediction_results,
        }
        df = pd.DataFrame(result_dic)
        df.to_csv(os.path.join(args.save_path,task_name + '_test_output.csv'))
        print("The predicion has been saved in " + os.path.join(args.save_path,task_name + '_test_output.csv'+"."))
        end_time = time()
        print("prediction time:",end_time-start_time)
    del net, dataloaders_dict
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='config/PTM_config_adapterH_prompt_test.yaml')
    parser.add_argument("--model_path", default='model/best_model_13ptm_final.pth')
    parser.add_argument("--data_path", default='data/Phosphorylation_S_sequence.fasta')
    parser.add_argument("--save_path", default='data')
    parser.add_argument("--PTM_type",default='Phosphorylation_S')

    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args, config_file)