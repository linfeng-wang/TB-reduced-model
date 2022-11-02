# training with split of data into training and validation using the same target stratification
# %% ###################################################################
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import util
import torchsummary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


drugs_sorted = res_all.isna().sum(axis=0).sort_values(ascending=False).index
least_common_drug_per_sample = pd.Series(pd.NA, index=res_all.index)
for idx, row in res_all.iterrows():
    for drug in drugs_sorted:
        if not pd.isna(row[drug]):
            least_common_drug_per_sample[idx] = drug
            break

RANDOM_SEED = 42
# use this now to create the stratified train/val split
train_idx, val_idx = train_test_split(
    least_common_drug_per_sample.index,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=least_common_drug_per_sample,
)


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


class OneHotSeqsDataset(torch.utils.data.Dataset): #? what's the difference between using inheritance and not?
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
            seqs_comb = self.seq_df.iloc[index].str.cat()
            res = self.res_df.iloc[index]
        elif isinstance(index, str):
            seqs_comb = self.seq_df.loc[index].str.cat()
            res = self.res_df.loc[index]
        else:
            raise ValueError(
                "Index needs to be an integer or a sample name present in the dataset"
            )
        return one_hot_torch(seqs_comb, dtype=self.one_hot_dtype), torch.tensor(res)

    def __len__(self):
        return self.res_df.shape[0]


train_dataset = OneHotSeqsDataset(seqs_df.loc[train_idx], res_all.loc[train_idx], one_hot_dtype=torch.float)
val_dataset = OneHotSeqsDataset(seqs_df.loc[val_idx], res_all.loc[val_idx], one_hot_dtype=torch.float)


# %% ###################################################################


def collate_padded_batch_old(batch):
    padded_seqs = nn.utils.rnn.pad_sequence([x[0].T for x in batch], batch_first=True)
    padded_batch = [(seq, x[1]) for seq, x in zip(padded_seqs, batch)]
    return torch.utils.data.default_collate(padded_batch) #? what does the default_collate function do


def collate_padded_batch(batch):
    # get max length of seqs in batch
    max_len = max([x[0].shape[1] for x in batch])
    return torch.utils.data.default_collate(
        [(F.pad(x[0], (0, max_len - x[0].shape[1])), x[1]) for x in batch] #how does F.pad work
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


m = Model(
    num_classes=len(DRUGS),
    num_dense_neurons=64,
    num_dense_layers=4,
    return_logits=True,
).to(device)
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

#%%
# from torchviz import make_dot
# x = torch.randn(1, 4, 56).to(device)
# y = m(x)
# make_dot(y, params=dict(list(m.named_parameters()))).render("cnn_torchviz", format="png")

# %% ###################################################################
# train now
# get_model first
num_filters=128
num_conv_layers=0
num_dense_neurons=64
num_dense_layers=0
return_logits=True
conv_dropout_rate= 0
step_size=10
gamma=0.5
N_epochs = 20
batch_size = 16

print(f'num_filters={num_filters} \n \
    num_conv_layers={num_conv_layers=} \n \
    num_dense_neurons={num_dense_neurons}\n \
    num_dense_layers={num_dense_layers}\n \
    return_logits={return_logits}\n \
    conv_dropout_rate= {conv_dropout_rate}\n \
    step_size={step_size}\n gamma={gamma}\n \
    N_epochs = {N_epochs} \n batch_size={batch_size}')
print('='*30)

m = Model(
    num_classes=len(DRUGS),
    num_filters=num_filters,
    num_conv_layers=num_conv_layers,
    num_dense_neurons=num_dense_neurons,
    num_dense_layers=num_dense_layers,
    return_logits=return_logits,
    conv_dropout_rate= conv_dropout_rate
).to(device)
# hyperparameters
optimizer = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-5)
#optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.3, dampening=0, weight_decay=0.3, nesterov=True, maximize=False, foreach=None)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
N_epochs = N_epochs
train_loss = []
# loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=128,
#     shuffle=True,
#     collate_fn=collate_padded_batch,
#     num_workers=4,
# )
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padded_batch,num_workers=4)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padded_batch,num_workers=4)

# training loop
train_loss_per_epoch = []
val_loss_per_epoch = []

accs = []
for epoch in range(N_epochs):
    m.train()
    loss_per_batch = []
    acc = AccumulatingMaskedAccuracy(drugs=DRUGS, device=device)
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred_logits = m(x_batch)
        loss = masked_BCE_from_logits(y_batch, y_pred_logits) #why are we not doing sigmoid before here
        loss.backward()
        optimizer.step()
        print(f"batch loss={round(loss.item(), 4)}", end="\r")
        loss_per_batch.append(loss.item())
        # update accuracy metric
        y_pred = torch.sigmoid(y_pred_logits)
        acc.update(y_batch, y_pred)
    epoch_loss = np.array(loss_per_batch).mean()
    train_loss_per_epoch.append(epoch_loss)
    accs.append(acc.values)
    # scheduler.step()
    print(f"epoch {epoch}: training loss={round(epoch_loss, 4)}")
    print(acc)
    m.eval()
    loss_per_batch = []
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred_logits = m(x_batch)
        loss = masked_BCE_from_logits(y_batch, y_pred_logits)
        loss_per_batch.append(loss.item())
    epoch_loss = np.array(loss_per_batch).mean()
    val_loss_per_epoch.append(epoch_loss)
    print(f"epoch {epoch}: validation loss={round(epoch_loss, 4)}")
    print('-----------------------')

    
# %% ###################################################################
fig, ax = plt.subplots()
x = np.arange(1, N_epochs+1, 1)
ax.plot(x, train_loss_per_epoch, label='Training')
ax.plot(x, val_loss_per_epoch, label='Validation')

ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, N_epochs+1, 10))
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig("/mnt/storageG1/lwang/TB-AMR-CNN/Julian/training_torch_simple_mask_copy_split_128f64n-s.png")

# %%
torch.save(m.state_dict(), '/mnt/storageG1/lwang/TB-AMR-CNN/Julian/training_torch_simple_mask_copy_split_model_128f64n-20e-spe30')

# %% ###################################################################
 