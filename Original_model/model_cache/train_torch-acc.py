#Model that has one sample per batch also outputing accuracy

#%%
#loading packages
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
from torchmetrics import Accuracy

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
import model_torch


model_torch = reload(model_torch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(88)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# load training data
seqs_df, res_all = util.load_data.get_main_dataset()
N_samples = seqs_df.shape[0]
DRUGS = util.DRUGS
assert set(DRUGS) == set(res_all.columns)
N_drugs = len(DRUGS)

# load the CRyPTIC samples as test data
seqs_cryptic, res_cryptic = util.load_data.get_cryptic_dataset()
# make sure the loci are in the same order as in the training data
seqs_cryptic = seqs_cryptic[seqs_df.columns]

# merging all columns of list into one
separator = "N"*30
seqs_df_agg =  seqs_df[list(seqs_df.columns)].agg(lambda x: separator.join(x.values), axis=1).T
res_all_combined = res_all.values.tolist()

seqs_cryptic_agg =  seqs_df[list(seqs_df.columns)].agg(lambda x: separator.join(x.values), axis=1).T
res_cryptic_combined = res_cryptic.values.tolist()


class RawReadDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y    
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

dataset = RawReadDataset(list(seqs_df_agg), res_all_combined) # dataset = CustomDataset(x_tensor, y_tensor)

#%%
def masked_BCE_from_logits(y_true, y_pred_logits):
    """
    Computes the BCE loss from logits and tolerates NaNs in `y_true`.
    """
    non_nan_ids = torch.argwhere(~torch.isnan(y_true))
    non_nan_ids = torch.flatten(non_nan_ids)
    y_pred_logits = torch.squeeze(y_pred_logits)
    y_true_non_nan = torch.index_select(y_true,0, non_nan_ids)
    y_pred_logits_non_nan = torch.index_select(y_pred_logits,0, non_nan_ids)
    # print(non_nan_ids)
    # print(y_true.size())
    # print(y_pred_logits.size())
    # print(y_true_non_nan)
    # print(y_pred_logits_non_nan)
    loss = torch.nn.MultiLabelSoftMarginLoss()
    return loss(y_true_non_nan, y_pred_logits_non_nan)

# def masked_BCE_from_logits(y_true, y_pred_logits):
#     """
#     Computes the BCE loss from logits and tolerates NaNs in `y_true`.
#     """
#     loss = torch.nn.MultiLabelSoftMarginLoss()
#     accuracy = Accuracy().to(device)
#     non_nan_ids = torch.argwhere(~torch.isnan(y_true))
#     non_nan_ids = torch.flatten(non_nan_ids)
#     y_pred_logits = torch.squeeze(y_pred_logits)
#     print("non_nan_ids:",non_nan_ids)
#     print("y_true.size:",y_true.size())
#     # print("y_pred_logits.size:",y_pred_logits.size())
#     # print(y_pred_logits)
#     y_true_non_nan = torch.index_select(y_true,0, non_nan_ids)
#     # print("y_true_non_nan:",y_true_non_nan)
#     y_pred_logits_non_nan = torch.index_select(y_pred_logits,0, non_nan_ids)
#     # print(y_pred_logits_non_nan)
#     # b_loss = torch.Tensor(b_loss)
#     # acc_list = torch.Tensor(acc_list)

#     return loss(y_true_non_nan, y_pred_logits_non_nan), accuracy(y_pred_logits_non_nan, y_true_non_nan)

train_dataset, val_dataset = random_split(dataset, [int(len(seqs_df_agg)*0.8), len(seqs_df_agg)-int(len(seqs_df_agg)*0.8)])

train_loader = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)

def one_hot_torch(seq):
    seq = torch.ByteTensor(list(bytes(seq, "utf-8")))
    acgt_bytes = torch.ByteTensor(list(bytes("ACGT", "utf-8")))
    arr = torch.zeros((len(seq), 4), dtype=torch.int8)
    arr[seq == acgt_bytes[0], 0] = 1
    arr[seq == acgt_bytes[1], 1] = 1
    arr[seq == acgt_bytes[2], 2] = 1
    arr[seq == acgt_bytes[3], 3] = 1
    return arr

#%%
# Testing conditions
# device = 'cpu'
# import os
# 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
# model_torch = reload(model_torch)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%%
#reloading model and running
# hyper-parameters
def make_train_step(model, loss_fn, optimizer, scheduler):
    def train_step(x, y):
        model.train()
        x = x.float()
        yhat = model(x)
        yhat = torch.squeeze(yhat)
        print(yhat.size())
        print(y.size())
        loss, acc = loss_fn(y, yhat)
        loss = torch.Tensor(loss)
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        #scheduler.step(0) # 0 used for learning rate policy 'plateau'
        optimizer.zero_grad()
        return loss.item(), acc
    return train_step

model = model_torch.raw_seq_model().to(device) # model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=64,min_lr=0.0001, verbose=True)
train_step = make_train_step(model, masked_BCE_from_logits, optimizer, scheduler)
n_epochs = 100
training_losses = []
validation_losses = []
lrs = []
training_acc = []
val_acc = []

# print(model.state_dict())

# training

for epoch in range(n_epochs):
    batch_losses = []
    batch_acc_train = []

    for x_batch, y_batch in train_loader:
        
        x_batch = one_hot_torch(x_batch[0])
        x_batch = x_batch[None, :]
        x_batch = x_batch.permute(0, 2, 1).to(device)
        y_batch = torch.Tensor(y_batch).to(device)
        print(x_batch.size())
        print(y_batch.size())

        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss)
        batch_acc_train.append(acc)

        # break
    #scheduler.step(0)
    epoch_acc_training = np.mean(batch_acc_train)
    training_acc.append(epoch_acc_training)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        batch_acc_val = []
        
        accuracy = Accuracy().to(device)
        for x_val, y_val in val_loader:
            x_val = one_hot_torch(x_val[0])
            x_val = x_val.float()
            x_val = x_val[None, :]
            x_val = x_val.permute(0, 2, 1).to(device)
            y_val = torch.Tensor(y_val).to(device)
            model.eval()
            yhat = model(x_val)
            val_loss, acc = masked_BCE_from_logits(y_val, yhat)#.item()
            val_losses.append(val_loss)
            batch_acc_val.append(acc)

            # break
        epoch_acc_val = np.mean(batch_acc_val)
        val_acc.append(epoch_acc_val)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    scheduler.step(validation_loss)
    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
    print(f"[{epoch+1}] Training Accuracy: {epoch_acc_training:.3f}\t Validation Accuracy: {epoch_acc_val:.3f}")
    print("="*20)
    # break
# print(model.state_dict())

#%%
#%%
torch.save(model.state_dict(), '/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/pytorch_model-one')

# model = model_torch_batch.raw_seq_model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
#%%
d = {'training_losses':training_losses,
     'validation_losses':validation_losses,
     'Training Accuracy': training_acc, 
     'Validation Accuracy': val_acc,
     "Learning rates": lrs}
history = pd.DataFrame.from_dict(d)
pd.DataFrame(history).to_csv("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/training-history-one.csv",index=False)
#%%
fig, ax = plt.subplots()
x = np.arange(1, n_epochs+1, 1)
ax.plot(x, history["training_losses"],label='Training')
ax.plot(x, history["validation_losses"],label='Validation')
ax.plot(x, history["Learning rates"],label='lrs')

ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, max(x)+1, 10))
# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/batch-training-loss-one.png")

fig, ax = plt.subplots()
x = np.arange(1, n_epochs+1, 1)
ax.plot(x, history["Training Accuracy"],label='Training')
ax.plot(x, history["Validation Accuracy"],label='Validation')
ax.plot(x, history["Learning rates"],label='lrs')

ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Accuracy")
ax.set_xticks(np.arange(0, max(x)+1, 10))
# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/batch-training-accuracy-one.png")


# %%