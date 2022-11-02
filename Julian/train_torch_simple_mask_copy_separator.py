# nan not droped and with all seq for input
# %% ###################################################################
from cmath import nan
from xml.sax import xmlreader
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import util
import torchsummary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% ###################################################################
# load training data
seqs_df, res_all = util.load_data.get_main_dataset()
N_samples = seqs_df.shape[0]
DRUGS = util.DRUGS
LOCI = seqs_df.columns
assert set(DRUGS) == set(res_all.columns)
N_drugs = len(DRUGS)

# load the CRyPTIC samples as test data
seqs_cryptic, res_cryptic = util.load_data.get_cryptic_dataset()
# make sure the loci are in the same order as in the training data
seqs_cryptic = seqs_cryptic[seqs_df.columns]

# %% ###################################################################


def one_hot_torch(seq: str, dtype=torch.int8):
    seq_bytes = torch.ByteTensor(list(bytes(seq, "utf-8")))
    acgt_bytes = torch.ByteTensor(list(bytes("ACGT", "utf-8")))
    arr = torch.zeros(4, (len(seq_bytes)), dtype=dtype)
    arr[0, seq_bytes == acgt_bytes[0]] = 1
    arr[1, seq_bytes == acgt_bytes[1]] = 1
    arr[2, seq_bytes == acgt_bytes[2]] = 1
    arr[3, seq_bytes == acgt_bytes[3]] = 1
    return arr


class OneHotSeqsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        seq_df,
        res_df,
        target_loci=LOCI,
        target_drugs=DRUGS,
        one_hot_dtype=torch.int8,
    ):
        self.seq_df = seq_df[target_loci]
        self.res_df = res_df[target_drugs]
        if not self.seq_df.index.equals(self.res_df.index):
            raise ValueError(
                "Indices of sequence and resistance dataframes don't match up"
            )
        self.one_hot_dtype = one_hot_dtype

    def __getitem__(self, index):
        """
        numerical index --> get `index`-th sample
        string index --> get sample with name `index`
        """
        if isinstance(index, int):
            seqs_comb = self.seq_df.iloc[index].str.cat(sep='X'*30)
            res = self.res_df.iloc[index]
        elif isinstance(index, str):
            seqs_comb = self.seq_df.loc[index].str.cat(sep='X'*30)
            res = self.res_df.loc[index]
        else:
            raise ValueError(
                "Index needs to be an integer or a sample name present in the dataset"
            )
        return one_hot_torch(seqs_comb, dtype=self.one_hot_dtype), torch.tensor(res)

    def __len__(self):
        return self.res_df.shape[0]


dataset = OneHotSeqsDataset(seqs_df, res_all, one_hot_dtype=torch.float)
# %% ###################################################################


# def collate_padded_batch_old(batch):
#     padded_seqs = nn.utils.rnn.pad_sequence([x[0].T for x in batch], batch_first=True)
#     padded_batch = [(seq, x[1]) for seq, x in zip(padded_seqs, batch)]
#     return torch.utils.data.default_collate(padded_batch)


def collate_padded_batch(batch):
    # get max length of seqs in batch
    max_len = max([x[0].shape[1] for x in batch])
    return torch.utils.data.default_collate(
        [(F.pad(x[0], (0, max_len - x[0].shape[1])), x[1]) for x in batch] #how does F.pad work
    )


loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, collate_fn=collate_padded_batch
)
# %% ###################################################################
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


# m = Model(
#     num_classes=len(DRUGS),
#     num_dense_neurons=64,
#     num_dense_layers=4,
#     return_logits=True,
# ).to(device)


# %% ###################################################################


def get_masked_loss(loss_fn):
    """
    Returns a loss function that ignores NaN values
    """

    def masked_loss(y_true, y_pred):
        non_nan_mask = ~y_true.isnan()
        y_true_non_nan = y_true[non_nan_mask]
        y_pred_non_nan = y_pred[non_nan_mask]
        return loss_fn(y_pred_non_nan, y_true_non_nan)

    return masked_loss


masked_BCE_from_logits = get_masked_loss(torch.nn.BCEWithLogitsLoss())


class AccumulatingMaskedAccuracy:
    def __init__(self, drugs, device="cpu"): 
        self.drugs = drugs
        self.accumulated_true = torch.zeros(len(drugs), device=device)
        self.accumulated_non_nan = torch.zeros(len(drugs), device=device)

    def update(self, y_true, y_pred):
        self.accumulated_true += (y_pred.round() == y_true).sum(axis=0).to(device)
        self.accumulated_non_nan += (~torch.isnan(y_true)).sum(axis=0).to(device)

    @property 
    def values(self):
        return pd.Series(
            (self.accumulated_true / self.accumulated_non_nan).detach().cpu(),
            index=self.drugs,
        )

    def __str__(self):
        return " --  ".join(
            f"{drug}: {round(accuracy, 3)}" for drug, accuracy in self.values.items()
        )

    def __repr__(self):
        return self.__str__()


# %% ###################################################################
# get_model first
m = Model(
    num_classes=len(DRUGS),
    num_filters=128,
    num_conv_layers=0,
    num_dense_neurons=64,
    num_dense_layers=0,
    return_logits=True,
).to(device)

# %% ###################################################################

import torchsummary
torchsummary.summary(m, (4, 50000))

from torchviz import make_dot
x = torch.randn(2, 4, 56).to(device)
y = m(x)
make_dot(y, params=dict(list(m.named_parameters()))).render("cnn_torchviz", format="png")
# %% ###################################################################


# hyperparameters
optimizer = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-5)
N_epochs = 20
train_loss = []
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_padded_batch,
    num_workers=4,
)

# %% ###################################################################
# training loop
m.train()
loss_per_epoch = []
accs = []
for epoch in range(N_epochs):
    loss_per_batch = []
    acc = AccumulatingMaskedAccuracy(drugs=DRUGS, device=device)
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred_logits = m(x_batch)
        loss = masked_BCE_from_logits(y_batch, y_pred_logits)
        loss.backward()
        optimizer.step()
        print(f"batch loss={round(loss.item(), 4)}", end="\r")
        loss_per_batch.append(loss.item())
        # update accuracy metric
        y_pred = torch.sigmoid(y_pred_logits)
        acc.update(y_batch, y_pred)
    epoch_loss = np.array(loss_per_batch).mean()
    loss_per_epoch.append(epoch_loss)
    accs.append(acc.values)
    print(f"epoch {epoch}: loss={round(epoch_loss, 4)}")
    print(acc)
    print('-----------------------')
# %% ###################################################################
fig, ax = plt.subplots()
x = np.arange(1, N_epochs+1, 1)
ax.plot(x, loss_per_epoch, label='Training')
ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, N_epochs+1, 10))
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/Julian/training_torch_simple_mask_copy_0c0d128f64n_d-sep.png")

# %%
torch.save(m.state_dict(), '/mnt/storageG1/lwang/TB-AMR-CNN/Julian/training_torch_simple_mask_copy_model_0c0d128f64n_d-separator30.pth')

# #%%
# #model evaluation
# missing_res = set(res_all.columns).difference(res_cryptic.columns)
# miss_df = pd.DataFrame(columns=missing_res)
# res_cryptic1 = pd.concat([res_cryptic, miss_df])
# res_cryptic1 = res_cryptic1[res_all.columns]

# #%%
# #creating eval Dataloader
# LOCI = seqs_cryptic.columns
# DRUGS = res_all.columns

# class OneHotSeqsDataset_val(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         seq_cryptic,
#         res_cryptic,
#         target_loci=LOCI,
#         target_drugs=DRUGS,
#         one_hot_dtype=torch.int8,
#     ):
#         self.seq_cryptic = seq_cryptic[target_loci]
#         self.res_cryptic = res_cryptic[target_drugs]
#         if not self.seq_cryptic.index.equals(self.res_cryptic.index):
#             raise ValueError(
#                 "Indices of sequence and resistance dataframes don't match up"
#             )
#         self.one_hot_dtype = one_hot_dtype

#     def __getitem__(self, index):
#         """
#         numerical index --> get `index`-th sample
#         string index --> get sample with name `index`
#         """
#         if isinstance(index, int):
#             seqs_comb = self.seq_cryptic.iloc[index].str.cat()
#             res = self.res_cryptic.iloc[index]
#         elif isinstance(index, str):
#             seqs_comb = self.seq_cryptic.loc[index].str.cat()
#             res = self.res_cryptic.loc[index]
#         else:
#             raise ValueError(
#                 "Index needs to be an integer or a sample name present in the dataset"
#             )
#         return one_hot_torch(seqs_comb, dtype=self.one_hot_dtype), torch.tensor(res)

#     def __len__(self):
#         return self.res_cryptic.shape[0]

# dataset = OneHotSeqsDataset_val(seqs_cryptic, res_cryptic1, one_hot_dtype=torch.float)

# loader = torch.utils.data.DataLoader(
#     dataset, batch_size=16, shuffle=True, collate_fn=collate_padded_batch
# )

# #%%
# device = 'cpu'

# m = Model(
#     num_classes=len(DRUGS),
#     num_filters=128,
#     num_conv_layers=2,
#     num_dense_neurons=64,
#     num_dense_layers=3,
#     return_logits=True,
# ).to(device)
# m.load_state_dict(torch.load('/mnt/storageG1/lwang/TB-AMR-CNN/Julian/training_torch_simple_mask_copy_model.pth'))
# m.eval()

# # loss_per_epoch = []
# # accs = []
# preds = []
# y_ = []

# # for epoch in range(N_epochs):
# #     loss_per_batch = []
# # acc = AccumulatingMaskedAccuracy(drugs=DRUGS, device=device)
# for x_batch, y_batch in tqdm(loader):
#     x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#     # optimizer.zero_grad()
#     y_pred_logits = m(x_batch)
#     # loss = masked_BCE_from_logits(y_batch, y_pred_logits)
#     # loss.backward()
#     # optimizer.step()
#     # print(f"batch loss={round(loss.item(), 4)}", end="\r")
#     # loss_per_batch.append(loss.item())
#     # update accuracy metric
#     y_pred = torch.sigmoid(y_pred_logits)
#     preds.append(y_pred)
#     y_.append(y_batch)
#     # y_pred_array = y_pred.detach().cpu().numpy()
#     # y_pred_array = y_pred_array.tolist()
#     # y_batch_auc = y_batch.detach().cpu().numpy().tolist()
#     # print(y_batch_auc)
#     # preds.extend(y_pred_array)
#     # acc.update(y_batch, y_pred)
# # epoch_loss = np.array(loss_per_batch).mean()
# # loss_per_epoch.append(epoch_loss)
# # accs.append(acc.values)
# # print(f"epoch {epoch}: loss={round(epoch_loss, 4)}")
# # print(acc)
# # print('-----------------------')

# # %%
# from torchmetrics import AUROC

# def auc_func(preds, y_stack):
#     res_index = [res_all.columns.tolist().index(x) for x in res_cryptic.columns.tolist()]
#     dr_cryptic = ['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','AMIKACIN','KANAMYCIN','MOXIFLOXACIN','LEVOFLOXACIN','ETHIONAMIDE']
#     y_stack =  torch.cat(y_, dim=0)
#     y_select =  y_stack[:, res_index]

#     preds_stack = torch.cat(preds, dim=0)
#     preds_select = preds_stack[:, res_index]
#     auc_dict = {}
#     for x in range(y_select.size()[1]):
#         ic(x)
#         no_nan = ~y_select[:,x].isnan()
#         pred_masked = preds_select[:,x][no_nan].to(device)
#         y_masked = y_select[:,x][no_nan].to(device)
#         # ic(pred_masked, y_masked)
#         # ic(len(pred_masked), len(y_masked))
#         auroc = AUROC(pos_label=1)
#         auc = auroc(pred_masked, y_masked.int())
#         auc = auc.cpu().numpy().tolist()
#         auc_dict[dr_cryptic[x]] = auc 
#     return auc_dict

# auc_dict = auc_func(preds, y_)
# %%
