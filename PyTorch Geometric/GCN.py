import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from torch_geometric.datasets import MoleculeNet


data = MoleculeNet(root=".", name="Tox21")


molecule = Chem.MolFromSmiles(data[0]["smiles"]) #for drawing molecule structures

# Implementing the GCN

import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
import warnings
from torch import autograd
from torchmetrics import AUROC
from torchmetrics import Accuracy
import pandas as pd
import seaborn as sns




embedding_size = 64
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 12)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling 
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # A final linear classifier
        out = self.out(hidden)
        return out, hidden
      
model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


warnings.filterwarnings("ignore")
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) 

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)

# Wrap data in training and test data loaders
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(data):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
    
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # Calculating the loss and gradients
      y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y)
      loss = loss_fn(pred, y)  
      if not torch.isnan(loss):   
        loss.backward()  
        # Update using the gradients
        optimizer.step()   

    return loss, embedding

print("Starting training...")
losses = []
for epoch in range(2000):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")
      
def test(data):
    # Enumerate over the data
    for batch in test_loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # Calculating the loss and gradients
      y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y)
      loss = loss_fn(pred, y)  
      if not torch.isnan(loss):   
        loss.backward()  
        # Update using the gradients
        optimizer.step()   
    return loss, embedding

print("Starting testing...")
losses_test = []
for epoch in range(2000):
    loss, h = test(data)
    losses_test.append(loss)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Test Loss {loss}")      

def evaluate_test():
  for batch in test_loader:
    batch.to(device)
    pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
    y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

  auroc = AUROC(num_classes=12)
  acc = Accuracy(num_classes=12)
  return auroc(pred,y).item(), acc(pred,y).item()

auroc, acc = evaluate_test()

print(f"ROC-AUC: {auroc}, accuracy: {acc}")

def evaluate_train():
  for batch in loader:
    batch.to(device)
    pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
    y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

  auroc = AUROC(num_classes=12)
  acc = Accuracy(num_classes=12)
  return auroc(pred,y).item(), acc(pred,y).item()

auroc_t, acc_t = evaluate_train()

print(f"ROC-AUC: {auroc_t}, accuracy: {acc_t}")


test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
    df = pd.DataFrame()
    y = torch.where(torch.isnan(test_batch.y), torch.zeros_like(test_batch.y), test_batch.y)
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])

plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
plt.set(xlim=(-0.5, 1.5))
plt.set(ylim=(-0.05, 0.1))
plt
