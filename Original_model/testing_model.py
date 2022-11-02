#%%
from array import array
from cmath import nan
from pyexpat import model
import statistics
from tkinter.ttk import Separator
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
from itertools import chain
from sklearn import metrics as met
import pickle
import icecream as ic

import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from importlib import reload
import util
from torchmetrics import Accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(88)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(os.path.basename(__file__))
#%%
# load training data
seqs_df, res_all = util.load_data.get_main_dataset()
N_samples = seqs_df.shape[0]
DRUGS = util.DRUGS
assert set(DRUGS) == set(res_all.columns)
N_drugs = len(DRUGS)

seqs_cryptic, res_cryptic = util.load_data.get_cryptic_dataset()
# make sure the loci are in the same order as in the training data
seqs_cryptic = seqs_cryptic[seqs_df.columns]

#merging all columns of list into one
#separator = "N"*30
separator = ""

# seqs_cryptic = seqs_cryptic[:100]
# res_cryptic = res_cryptic[:100]

seqs_cryptic_agg =  seqs_cryptic[list(seqs_df.columns)].agg(lambda x: separator.join(x.values), axis=1).T
#res_cryptic_combined = res_cryptic.values.tolist()
res_cryptic_combined = res_cryptic["ISONIAZID"].values.tolist()

class RawReadDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y    
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

dataset = RawReadDataset(seqs_cryptic_agg, res_cryptic_combined) # dataset = CustomDataset(x_tensor, y_tensor)

def masked_BCE_from_logits(y_true, y_pred_logits):
    """
    Computes the BCE loss from logits and tolerates NaNs in `y_true`.
    """ 
    accuracy = Accuracy().to(device)
    bce_logits = torch.nn.BCEWithLogitsLoss()
    # print("y_true", y_true.size())
    non_nan_mask = ~y_true.isnan()
    # print("non_nan_mask:", non_nan_mask.size())
    y_true_non_nan = y_true[non_nan_mask]
    # print("y_true_non_nan:", y_true_non_nan.size())
    y_pred_logits_non_nan = y_pred_logits[non_nan_mask]
    # print("y_pred_logits_non_nan:", y_pred_logits_non_nan.size())
    y_pred_logits_non_nan = y_pred_logits_non_nan.squeeze(dim = -1)
    return bce_logits(y_pred_logits_non_nan, y_true_non_nan), accuracy(y_pred_logits_non_nan, y_true_non_nan.int())

# train_dataset, val_dataset = random_split(dataset, [int(len(seqs_df_agg)*0.8), len(seqs_df_agg)-int(len(seqs_df_agg)*0.8)])
test_loader = DataLoader(dataset=dataset, batch_size=1024)
# val_loader = DataLoader(dataset=val_dataset, batch_size=32)

def one_hot_torch(seq):
    oh = []
    for sample in seq:
        sample = torch.ByteTensor(list(bytes(sample, "utf-8")))
        acgt_bytes = torch.ByteTensor(list(bytes("ACGT", "utf-8")))
        arr = torch.zeros((len(sample), 4), dtype=torch.int8)
        arr[sample == acgt_bytes[0], 0] = 1
        arr[sample == acgt_bytes[1], 1] = 1
        arr[sample == acgt_bytes[2], 2] = 1
        arr[sample == acgt_bytes[3], 3] = 1
        oh.append(arr)
    return oh

def my_padding(seq_tuple):
    list_x_ = list(seq_tuple)
    max_len = len(max(list_x_, key=len))
    for i, x in enumerate(list_x_):
        list_x_[i] = x + "N"*(max_len-len(x))
    return list_x_

#%%
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

import model_torch_simple
model = model_torch_simple.raw_seq_model()
PATH = "/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/pytorch_model-simple"
model.load_state_dict(torch.load(PATH))
model = model.to(device)
#%%
model.eval()

val_losses = []
batch_acc_val = []
for x_val, y_val in test_loader:
    x_val = my_padding(x_val)
    x_val = one_hot_torch(x_val)
    x_val = torch.stack(x_val, dim=1).to(device)
    x_val = x_val.float()
    x_val = x_val.permute(1, 2, 0).to(device)
    # y_val = torch.stack(y_val, dim=1).to(device)
    yhat = model(x_val)
    val_loss, acc = masked_BCE_from_logits(y_val, yhat)#.item()
    val_losses.append(val_loss.item())
    batch_acc_val.append(acc.item())
# %%
print("loss:",np.mean(val_losses))
print("accuracy:",np.mean(batch_acc_val))
# %%
len(val_losses)
# %%
