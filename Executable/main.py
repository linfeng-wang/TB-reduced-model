#%%

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics as met
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from importlib import reload
from torchmetrics import Accuracy
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from torch.utils.data import default_convert
import argparse

#%%

parser = argparse.ArgumentParser(
            description='''TB drug resistance prediction for AMIKACIN, CAPREOMYCIN, CIPROFLOXACIN, ETHAMBUTOL, ETHIONAMIDE, \
                                                ISONIAZID, KANAMYCIN, LEVOFLOXACIN, MOXIFLOXACIN, OFLOXACIN, \
                                                PYRAZINAMIDE, RIFAMPICIN, STREPTOMYCIN from consensus sequence.''')

parser.add_argument("-i", "--input", 
                    help='csv file containing one-hot encoded consensus sequence [required]', 
                    type=str,
                    metavar="FILE",required = True)

parser.add_argument("-o", "--output", 
                    help='output path into csv format with drug resistance type as header', 
                    type=str,
                    metavar="FILE",required = False)

parser.add_argument("-v", "--verbose", 
                    help='Output prediction in as text in terminal', required = False,  action='store_true')

args = parser.parse_args()

#%%

def one_hot_torch(seq: str, dtype=torch.int8):
    seq_bytes = torch.ByteTensor(list(bytes(seq, "utf-8")))
    acgt_bytes = torch.ByteTensor(list(bytes("ACGT", "utf-8")))
    arr = torch.zeros(4, (len(seq_bytes)), dtype=dtype)
    arr[0, seq_bytes == acgt_bytes[0]] = 1
    arr[1, seq_bytes == acgt_bytes[1]] = 1
    arr[2, seq_bytes == acgt_bytes[2]] = 1
    arr[3, seq_bytes == acgt_bytes[3]] = 1
    return arr

def collate_padded_batch(batch):
    # get max length of seqs in batch
    max_len = max([x[0].shape[1] for x in batch])
    return torch.utils.data.default_collate(
        [(F.pad(x[0], (0, max_len - x[0].shape[1])), x[1]) for x in batch] #how does F.pad work
    )

class Model(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=1,
        num_filters=64,
        filter_length=25,
        num_conv_layers=2,
        num_dense_neurons=256,
        num_dense_layers=2,
        conv_dropout_rate=0.0,
        dense_dropout_rate=0.2,
        return_logits=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.conv_dropout_rate = conv_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.return_logits = return_logits
        # now define the actual model
        self.feature_extraction_layer = self._conv_layer(
            in_channels, num_filters, filter_length
        )
        self.conv_layers = nn.ModuleList(
            self._conv_layer(num_filters, num_filters, 3)
            for _ in range(num_conv_layers)
        )
        self.dense_layers = nn.ModuleList(
            self._dense_layer(input_dim, num_dense_neurons)
            for input_dim in [num_filters]
            + [num_dense_neurons] * (num_dense_layers - 1) #how does this work?
        )
        self.prediction_layer = (
            nn.Linear(num_dense_neurons, num_classes)
            if return_logits
            else nn.Sequential(nn.Linear(num_dense_neurons, num_classes), nn.Sigmoid()) #difference between sequential and nn.moduleList?
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Dropout(p=self.conv_dropout_rate),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def _dense_layer(self, n_in, n_out):
        return nn.Sequential(
            nn.Dropout(p=self.dense_dropout_rate),
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
        )

    def forward(self, x):
        # first pass over input
        # print(x.size())
        x = self.feature_extraction_layer(x)
        # conv layers
        for layer in self.conv_layers:
            x = layer(x)
        # global max pool 1D
        x = torch.max(x, dim=-1).values
        # fully connected layers
        for layer in self.dense_layers:
            x = layer(x)
        x = self.prediction_layer(x)
        return x

#%%
outcome = []
device = "cuda" if torch.cuda.is_available() else "cpu"

#running model
full_seq = pd.read_csv(f'{args.input}', header=0)
# full_seq = pd.read_csv('test_input_ERR757145.csv', header=0) 

full_seq_input = torch.tensor(full_seq.values).T.to(device).float()
full_seq_input = full_seq_input[None, :]

m = Model(
num_classes=13,
num_filters=128,
num_conv_layers=0,
num_dense_neurons=64,
num_dense_layers=0,
return_logits=True,
).to(device)

m.load_state_dict(torch.load('training_torch_simple_mask_copy_split_model_128f64n-spe30-rand5-100e'))
m.eval()

y_pred_logits = m(full_seq_input)
y_pred = torch.sigmoid(y_pred_logits)
y_pred = y_pred.detach().cpu()

# %%
dr = ['AMIKACIN', 'CAPREOMYCIN', 'CIPROFLOXACIN', 'ETHAMBUTOL', 'ETHIONAMIDE',
       'ISONIAZID', 'KANAMYCIN', 'LEVOFLOXACIN', 'MOXIFLOXACIN', 'OFLOXACIN',
       'PYRAZINAMIDE', 'RIFAMPICIN', 'STREPTOMYCIN']
pred = torch.round(y_pred)
results = pred.numpy().tolist()[0]
df_ = pd.DataFrame(columns=dr)
df_.loc[0] = results

# %%
if args.verbose:
    print('='*30)
    print(args.input)
    print('*'*10)
    print(df_)
    print('*'*10)

if args.output:
    df_.to_csv(f'{args.output}', index=False, index_label=False)

#test running
#python main.py -i test_input_ERR757145.csv -o test_out.csv -v
    
# %%
