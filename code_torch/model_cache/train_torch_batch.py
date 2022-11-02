#training with >1 sample per batch
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
import model_torch_batch
from torchmetrics import Accuracy

model_torch_batch = reload(model_torch_batch)

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
# def masked_BCE_from_logits(y_true, y_pred_logits):
#     """
#     Computes the BCE loss from logits and tolerates NaNs in `y_true`.
#     """
#     b_loss = 0
#     #loss = torch.nn.MultiLabelSoftMarginLoss()
#     loss = torch.nn.BCEWithLogitsLoss()
#     acc_list = []
#     accuracy = Accuracy().to(device)
#     for i in range(0,len(y_true)): # this loop is done because after removal of nan, the tensor would have ben rugged
#         non_nan_ids = torch.argwhere(~torch.isnan(y_true[i]))
#         non_nan_ids = torch.flatten(non_nan_ids)
#         # print("non_nan_ids:",non_nan_ids)
#         # print("y_true.size:",y_true.size())
#         # print("y_pred_logits.size:",y_pred_logits.size())
#         # print(y_pred_logits)
#         y_true_non_nan = torch.index_select(y_true[i],0, non_nan_ids)
#         # print("y_true_non_nan:",y_true_non_nan)

#         y_pred_logits_non_nan = torch.index_select(y_pred_logits[i],0, non_nan_ids)
#         # print(y_pred_logits_non_nan)
        
#         loss_value = loss(y_pred_logits_non_nan, y_true_non_nan)
#         # b_loss.append(loss_value)
#         y_true_non_nan = y_true_non_nan.int()
#         acc_list.append(accuracy(y_pred_logits_non_nan, y_true_non_nan))
#         b_loss += loss_value
#     # b_loss = torch.Tensor(b_loss)
#     # print(torch.nn.MarginRankingLossExp()(b_loss))
#     acc_list = torch.Tensor(acc_list)

#     return b_loss/len(y_true), torch.mean(acc_list)

#%%
def masked_BCE_from_logits(y_true, y_pred_logits):
    """
    Computes the BCE loss from logits and tolerates NaNs in `y_true`.
    """
    bce_logits = torch.nn.BCEWithLogitsLoss()
    # print("y_true", y_true.size())
    non_nan_mask = ~y_true.isnan()
    # print("non_nan_mask:", non_nan_mask.size())
    y_true_non_nan = y_true[non_nan_mask]
    # print("y_true_non_nan:", y_true_non_nan.size())
    y_pred_logits_non_nan = y_pred_logits[non_nan_mask]
    # print("y_pred_logits_non_nan:", y_pred_logits_non_nan.size())

    return bce_logits(y_pred_logits_non_nan, y_true_non_nan), 0



train_dataset, val_dataset = random_split(dataset, [int(len(seqs_df_agg)*0.8), len(seqs_df_agg)-int(len(seqs_df_agg)*0.8)])

train_loader = DataLoader(dataset=train_dataset, batch_size=32)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)


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
#reloading model and running
#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_torch_batch = reload(model_torch_batch)

# hyper-parameters
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        x = x.float()
        yhat = model(x)
        loss, acc = loss_fn(y, yhat)
        loss.requires_grad_()
        loss.backward()
        optimizer.step()
        # scheduler.step(0)
        optimizer.zero_grad()
        return loss.item(), acc#.item()
    return train_step

model = model_torch_batch.raw_seq_model().to(device) # model = nn.Sequential(nn.Linear(1, 1)).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, maximize=False, foreach=None)
train_step = make_train_step(model, masked_BCE_from_logits, optimizer)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,min_lr=0.0001, verbose=True)
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
    training_batch = 0
    for x_batch, y_batch in train_loader:
        x_batch = my_padding(x_batch)
        x_batch = one_hot_torch(x_batch)
        x_batch = torch.stack(x_batch, dim=1).to(device)
        x_batch = x_batch.permute(1, 2, 0).to(device)
        y_batch = torch.stack(y_batch, dim=1).to(device)
        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss)
        batch_acc_train.append(acc)
        
        # print(f"training batch: {training_batch}")
        # training_batch += 1
        #break
    epoch_acc_training = np.mean(batch_acc_train)
    training_acc.append(epoch_acc_training)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)
    lrs.append(optimizer.param_groups[0]["lr"])

    with torch.no_grad():
        val_losses = []
        batch_acc_val = []
        val_batch = 0
        for x_val, y_val in val_loader:
            x_val = my_padding(x_val)
            x_val = one_hot_torch(x_val)
            x_val = torch.stack(x_val, dim=1).to(device)
            x_val = x_val.float()
            x_val = x_val.permute(1, 2, 0).to(device)

            y_val = torch.stack(y_val, dim=1).to(device)

            model.eval()
            yhat = model(x_val)
            val_loss, acc = masked_BCE_from_logits(y_val, yhat)
            val_losses.append(val_loss.item())
            batch_acc_val.append(acc)
            
            # print(f"val batch: {val_batch}")
            # val_batch += 1
            #break
        #scheduler.step(val_loss)
        epoch_acc_val = np.mean(batch_acc_val)
        val_acc.append(epoch_acc_val)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
    print(f"[{epoch+1}] Training Accuracy: {epoch_acc_training:.3f}\t Validation Accuracy: {epoch_acc_val:.3f}")
    print("="*20)
    # break
print(model.state_dict())
#%%
torch.save(model.state_dict(), '/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/pytorch_model-batch-acc')

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
pd.DataFrame(history).to_csv("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/training-history-batch-acc.csv",index=False)
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

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/batch-training-loss-batch-acc.png")

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

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/batch-training-accuracy-batch-acc.png")


# %%
# %%
bla = 'blu'
# %%


def func():
    raise ValueError()

    
func()

#%%
x, y = next(iter(train_loader))
# %%
y
# %%
