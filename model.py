import torch
from peft import LoraConfig, get_peft_model
import esm_adapterH
import numpy as np
from torch import nn
import os
from esm_adapterH.prompt_tuning import PrefixTuning
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_task_embedding(configs):

    task_embeddings_dic= {}
    output_data_folder=configs.encoder.prompt.task_token_path

    if configs.tasks.Phosphorylation_S == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_S_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_S, task_embedding)

    if configs.tasks.Phosphorylation_T == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_T_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_T, task_embedding)

    if configs.tasks.Phosphorylation_Y == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_Y_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_Y, task_embedding)

    if configs.tasks.Ubiquitination_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Ubiquitination_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Ubiquitination_K, task_embedding)

    if configs.tasks.Acetylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Acetylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Acetylation_K, task_embedding)

    if configs.tasks.OlinkedGlycosylation_S == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "OlinkedGlycosylation_S_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.OlinkedGlycosylation_S, task_embedding)

    if configs.tasks.Methylation_R == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Methylation_R_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Methylation_R, task_embedding)

    if configs.tasks.NlinkedGlycosylation_N == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "NlinkedGlycosylation_N_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.NlinkedGlycosylation_N, task_embedding)

    if configs.tasks.OlinkedGlycosylation_T == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "OlinkedGlycosylation_T_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.OlinkedGlycosylation_T, task_embedding)

    if configs.tasks.Methylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Methylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Methylation_K, task_embedding)

    if configs.tasks.Palmitoylation_C == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Palmitoylation_C_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Palmitoylation_C, task_embedding)

    if configs.tasks.Sumoylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Sumoylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Sumoylation_K, task_embedding)

    if configs.tasks.Succinylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Succinylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Succinylation_K, task_embedding)

    return task_embeddings_dic

def prepare_adapter_h_model(configs):

    adapter_args = configs.encoder.adapter_h
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm_adapterH.pretrained, model_name, None)
    model, alphabet = model_constructor(adapter_args)
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    if configs.encoder.prompt.enable:
        if not hasattr(configs.encoder.prompt, "num_tasks"):
            configs.encoder.prompt.num_tasks = 1

        task_embedding_dic = get_task_embedding(configs)

        model.prefix_module = PrefixTuning(configs, task_embedding_dic, model, prompt_len=configs.encoder.prompt.prompt_len,
                                           prompt_layer_indices=configs.encoder.prompt.prompt_layer_indices,
                                           num_tasks=configs.encoder.prompt.num_tasks
                                           )
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    if configs.encoder.adapter_h.enable:
      if not isinstance(configs.encoder.adapter_h.freeze_adapter_layers, list):
        configs.encoder.adapter_h.freeze_adapter_layers = [configs.encoder.adapter_h.freeze_adapter_layers]
    
    if configs.encoder.fine_tune.enable:
      if not isinstance(configs.encoder.fine_tune.freeze_adapter_layers, list):
        configs.encoder.fine_tune.freeze_adapter_layers = [configs.encoder.fine_tune.freeze_adapter_layers]
    
    if configs.encoder.lora.enable:

        if hasattr(configs.encoder.lora,"lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)


    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    

    if configs.encoder.adapter_h.enable:
      for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
        if not value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    param.requires_grad = True
    
    if configs.encoder.fine_tune.enable:
       for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
        if value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    print("freeze adapter in fine-tune")
                    param.requires_grad = False


    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True


    if configs.encoder.prompt.enable:
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    return model, alphabet


class MoBYMLP_multihead(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2,num_tasks=1):
        super(MoBYMLP_multihead, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.task_heads = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_tasks)])


    def forward(self, x):
        batch_size,seqlen,features_dim = x.shape
        x = x.reshape(-1,features_dim)
        x = self.linear_hidden(x)
        task_outputs = [head(x).reshape(batch_size, seqlen, -1) for head in self.task_heads]  # List of outputs for each task

        return task_outputs


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, configs,stride=1,padding="same"):
        super(MultiScaleCNN, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, configs.projector.out_channels, configs.projector.kernel_sizes[i], stride, padding) for i in range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm1 = nn.BatchNorm1d(configs.projector.out_channels * len(configs.projector.kernel_sizes))  # BatchNorm after the first Conv layer
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(configs.projector.out_channels * len(configs.projector.kernel_sizes), configs.projector.out_channels,  configs.projector.kernel_sizes[i], stride, padding) for i in range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm2 = nn.BatchNorm1d(configs.projector.out_channels * len(configs.projector.kernel_sizes))  # BatchNorm after the second Conv layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=configs.projector.droprate)

        if configs.projector.if_multihead==True:
            self.fc_multiclass = MoBYMLP_multihead(
                in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                inner_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers,num_tasks=configs.encoder.prompt.num_tasks)

        self.configs=configs

    def forward(self, x):
        x=x.permute(0,2,1)

        conv_results = [conv_layer(x) for conv_layer in self.conv_layers]
        x = torch.cat(conv_results, dim=1)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        conv_results2 = [conv_layer(x) for conv_layer in self.conv_layers2]
        x = torch.cat(conv_results2, dim=1)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x_contact = x.permute(0, 2, 1)  # batch, length, out_channels
        x = self.fc_multiclass(x_contact)  # batch, length, output_dim
        return x

class EncoderSSPTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs)
        self.configs=configs
        # extract the embedding size
        if  self.configs.encoder.model_name.startswith("esmc"):
            mlp_input_dim = self.esm2.embed.embedding_dim
        else:
            mlp_input_dim = self.esm2.embed_dim

        if configs.projector.projector_type=='CNN':

            in_channels= mlp_input_dim
            self.mlp = MultiScaleCNN(in_channels,configs)

        self.configs = configs

    def forward(self, x, task_ids):
        features = self.esm2(x['input_ids'].to(device),repr_layers=[self.esm2.num_layers],task_ids=task_ids,configs=self.configs)['representations'][self.esm2.num_layers]

        if self.configs.encoder.prompt.if_pass_to_MHA:
            c=self.mlp(torch.concat([features[:, 0:500, :],features[:, 501:-1, :]],dim=1))
        else:
            c = self.mlp(features[:, 1:-1, :])
        return c


def prepare_models_secondary_structure_ptm(configs):
    encoder = EncoderSSPTM(configs=configs)

    return encoder



