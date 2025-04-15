import torch
import yaml
from torch import nn
from torch.nn import functional as F
from collections.abc import Sequence
from transformers import EsmModel, T5Tokenizer, T5Model
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import load_configs, get_dummy_logging
import esm_adapterH
import esm
import numpy as np
import copy
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from typing import Mapping, Optional, Tuple, Any, Union
from torch import nn, Tensor
import os

# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig



from esm_adapterH.prompt_tuning import PrefixTuning
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def verify_data_types(model, logging=None):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        if logging:
           logging.info(f"{k}, {v}, {v / total}")



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

        task_embedding_dic = get_task_embedding(configs)    ##get pretrain task token

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

    #     model.prefix_module.prompt_layer_dict["layer_0"]["0"].requires_grad=False
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
            #modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config)

        # verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    
    
    # only freeze all the parameters once at the beginning. then open some layers later
    #only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
      for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
        if not value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    param.requires_grad = True
    
    # only freeze all the parameters once at the beginning. then open some layers later,but because
    # of fine_tune, adapter layers might be tunable.
    #change on 1/15/2024 not need to use freeze_adapter_layers to control fine-tune part! use another parameter instead and must after setting of freeze_adapter_layers
    if configs.encoder.fine_tune.enable: #only see fine_tune.freeze_adapter_layers when fine-tune is available
       for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
        if value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    print("freeze adapter in fine-tune")
                    param.requires_grad = False
    #"""

    
    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    # if configs.encoder.prompt.enable:
    #     for param in model.prefix_module.parameters():
    #         param.requires_grad = True
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
            linear_hidden.append(nn.ReLU(inplace=True)) #relu cannot be used with sigmoid!!! smallest will be 0.5?
        self.linear_hidden = nn.Sequential(*linear_hidden)
        #
        # self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
        #                             out_dim) if num_layers >= 1 else nn.Identity()

        self.task_heads = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_tasks)])
        # self.task_weights = task_weights  # A list or tensor of weights for each task

    def forward(self, x):
        #print("mlp forward")
        #print(x.shape)  #[128,512,100]
        batch_size,seqlen,features_dim = x.shape
        x = x.reshape(-1,features_dim)
        x = self.linear_hidden(x)
        task_outputs = [head(x).reshape(batch_size, seqlen, -1) for head in self.task_heads]  # List of outputs for each task


        # x = self.linear_out(x)
        # x = x.reshape(batch_size,seqlen,-1)

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
        # self.fc_shared = nn.Linear(out_channels * len(kernel_sizes), inner_linear_dim)  # Shared fully connected layer
        # self.fc_multiclass = nn.Linear(inner_linear_dim, output_dim)  # Output layer for multi-class classification
        # use on 7.25.2024 4:19
        if configs.projector.if_multihead==True:
            self.fc_multiclass = MoBYMLP_multihead(
                in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                inner_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers,num_tasks=configs.encoder.prompt.num_tasks)

        self.configs=configs

    def forward(self, x):

        #x  batch length input channel
        x=x.permute(0,2,1)
        # x=x.unsqueeze(2)

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
        # x = self.fc_shared(x)
        # print(x.shape)
        x = self.fc_multiclass(x_contact)  # batch, length, output_dim
        return x


# Define Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        # hidden = F.softmax(hidden, dim=-1)
        return hidden


# def prepare_configs_mergedESM2(configs):
#         merged_configs = copy.deepcopy(configs)
#         #if has tune_embedding in merge2ESM2 use this specific config, if not, share with original configs_all
#         if hasattr(configs.encoder.merge2ESM2,"tune_embedding"):
#             merged_configs.encoder.tune_embedding = configs.encoder.merge2ESM2.tune_embedding
#         if hasattr(configs.encoder.merge2ESM2,"fine_tune"):
#            merged_configs.encoder.fine_tune = configs.encoder.merge2ESM2.fine_tune
#         if hasattr(configs.encoder.merge2ESM2,"lora"):
#            merged_configs.encoder.lora = configs.encoder.merge2ESM2.lora
#         if hasattr(configs.encoder.merge2ESM2,"adapter_h"):
#            merged_configs.encoder.adapter_h = configs.encoder.merge2ESM2.adapter_h
#
#         return merged_configs



class EncoderSSPTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs)
        else:
            # self.esm2, self.alphabet = prepare_esm_model(configs, logging)
            self.esm2 = prepare_esm_model(configs)
        self.configs=configs
        # extract the embedding size
        if  self.configs.encoder.model_name.startswith("esmc"):
            mlp_input_dim = self.esm2.embed.embedding_dim
        else:
            mlp_input_dim = self.esm2.embed_dim

        if configs.projector.projector_type=='MLP':

            mlp_hidden_dim = configs.encoder.mlp_hidden_dim
            hidden_dims = [mlp_hidden_dim] * (configs.encoder.mlp_layer_num - 1)
            self.mlp = MultiLayerPerceptron(mlp_input_dim, hidden_dims + [configs.encoder.num_classes], batch_norm=False,
                                            dropout=configs.encoder.head_dropout)

        elif configs.projector.projector_type=='CNN':

            in_channels= mlp_input_dim
            self.mlp = MultiScaleCNN(in_channels,configs)

        # self.device = device
        self.configs = configs

    def forward(self, x, task_ids):



        features = self.esm2(x['input_ids'].to(device),repr_layers=[self.esm2.num_layers],task_ids=task_ids,configs=self.configs)['representations'][self.esm2.num_layers]

        if self.configs.encoder.prompt.if_pass_to_MHA:
            # c = self.mlp(torch.concat([features[:, 0:500, :],features[:, 501:-1, :]],dim=1))
            c=self.mlp(torch.concat([features[:, 0:500, :],features[:, 501:-1, :]],dim=1))
        else:
            c = self.mlp(features[:, 1:-1, :])
        return c


# def get_nb_trainable_parameters(model):
#     r"""
#     Returns the number of trainable parameters and number of all parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         num_params = param.numel()
#         # if using DS Zero 3 and the weights are initialized empty
#         if num_params == 0 and hasattr(param, "ds_numel"):
#             num_params = param.ds_numel
#
#             # Due to the design of 4bit linear layers from bitsandbytes
#         # one needs to multiply the number of parameters by 2 to get
#         # the correct number of parameters
#         if param.__class__.__name__ == "Params4bit":
#             num_params = num_params * 2
#
#         all_param += num_params
#         if param.requires_grad:
#             trainable_params += num_params
#
#     return trainable_params, all_param
#
#
# def print_trainable_parameters(model, logging):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params, all_param = get_nb_trainable_parameters(model)
#     logging.info(
#         f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
#     )


# def prepare_models(configs, logging):
#     """
#     Prepare the encoder model.
#
#     Args:
#         configs: A python box object containing the configuration options.
#         logging: The logging object.
#
#     Returns:
#         The encoder model.
#     """
#     # Prepare the encoder.
#     encoder = Encoder(logging=logging, configs=configs)
#     print_trainable_parameters(encoder, logging)
#     logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))
#
#     return encoder

# def prepare_models_merge(configs, logging):
#     """
#     Prepare the encoder model.
#
#     Args:
#         configs: A python box object containing the configuration options.
#         logging: The logging object.
#
#     Returns:
#         The encoder model.
#     """
#     # Prepare the encoder.
#     encoder = Encoder_merge(logging=logging, configs=configs)
#     print_trainable_parameters(encoder, logging)
#     logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))
#
#     return encoder

def prepare_models_secondary_structure_ptm(configs):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = EncoderSSPTM(configs=configs)
    # print_trainable_parameters(encoder, logging)
    # logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder



# if __name__ == '__main__':
#     # For test model and its modules
#     config_path = './config.yaml'
#     with open(config_path) as file:
#         configs_dict = yaml.full_load(file)
#
#     configs_file = load_configs(configs_dict)
#
#     dummy_logging = get_dummy_logging()
#
#     encoder_model = prepare_models(configs_file, dummy_logging)
#     input_tensor = torch.randint(high=30, low=0, size=(2, 1024), dtype=torch.int64)
#
#     sample = {'input_ids': input_tensor, 'attention_mask': torch.ones(input_tensor.shape)}
#     output = encoder_model(sample)
#     print(output.shape)
#     print('done')


#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel
#
# class MultitaskBERT(nn.Module):
#     def __init__(self, num_tasks, hidden_size, task_heads_output_dim, task_weights):
#         super(MultitaskBERT, self).__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.task_heads = nn.ModuleList([nn.Linear(hidden_size, task_heads_output_dim) for _ in range(num_tasks)])
#         self.task_weights = task_weights  # A list or tensor of weights for each task
#
#     def forward(self, input_ids, attention_mask, task_id):
#         # Shared BERT encoder
#         bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = bert_output.pooler_output  # Shape: [batch_size, hidden_size]
#
#         # Get task-specific outputs
#         task_outputs = [head(pooled_output) for head in self.task_heads]  # List of outputs for each task
#         return task_outputs
#
#     def compute_weighted_loss(self, task_outputs, labels, task_id):
#         # task_outputs: List of [batch_size, num_labels] logits for each task
#         # labels: [batch_size] true labels
#         # task_id: [batch_size] task ids for each example
#
#         batch_size = labels.size(0)
#         total_loss = 0.0
#
#         # Compute weighted loss for each example in the batch
#         for i in range(batch_size):
#             example_task_id = task_id[i].item()  # Get the task ID for this example
#             logits = task_outputs[example_task_id][i]  # Get the output for the correct task
#             example_loss = F.cross_entropy(logits.unsqueeze(0), labels[i].unsqueeze(0), reduction='mean')
#             weighted_loss = example_loss * self.task_weights[example_task_id]
#             total_loss += weighted_loss
#
#         return total_loss / batch_size  # Average loss over the batch
#
# # Usage Example
# # num_tasks = 2, hidden_size = 768 (BERT's hidden size), task_heads_output_dim = num_classes for each task
# num_tasks = 2
# hidden_size = 768
# task_heads_output_dim = 2  # Assuming binary classification for simplicity
# task_weights = torch.tensor([1.0, 0.5])  # Example weights for tasks
#
# model = MultitaskBERT(num_tasks, hidden_size, task_heads_output_dim, task_weights)
#
# # Assume input_ids, attention_mask, labels, and task_id are batch tensors
# # input_ids, attention_mask shape: [batch_size, seq_length]
# # labels, task_id shape: [batch_size]
# task_outputs = model(input_ids, attention_mask, task_id)
# loss = model.compute_weighted_loss(task_outputs, labels, task_id)
#
#
# import torch
# from torch.utils.data import Dataset, DataLoader
#
# class MultitaskDataset(Dataset):
#     def __init__(self, data, labels, task_ids):
#         """
#         data: List of input_ids for each example.
#         labels: List of labels for each example.
#         task_ids: List of task IDs for each example, representing the task they belong to.
#         """
#         self.data = data
#         self.labels = labels
#         self.task_ids = task_ids
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return {
#             'input_ids': self.data[idx],
#             'attention_mask': torch.ones_like(self.data[idx]),  # Example attention mask
#             'labels': self.labels[idx],
#             'task_id': self.task_ids[idx]
#         }
#
# # Example usage of the dataset and DataLoader
# # Assume data, labels, and task_ids are pre-defined lists
# data = [torch.tensor([101, 1045, 2572, 1037, 2307, 3185, 102])] * 10  # Dummy tokenized inputs
# labels = [torch.tensor(1)] * 10  # Dummy labels
# task_ids = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # Example task IDs for each example
#
# dataset = MultitaskDataset(data, labels, task_ids)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# # Iterating over the DataLoader
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     labels = batch['labels']
#     task_id = batch['task_id']
#     # Now `task_id` is available for each example in the batch
