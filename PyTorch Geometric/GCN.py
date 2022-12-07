import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from torch_geometric.datasets import MoleculeNet
import numpy as np
import time
from datetime import timedelta
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


data = MoleculeNet(root=".", name="Tox21")


molecule = Chem.MolFromSmiles(data[0]["smiles"]) #for drawing molecule structures


# Implementing the GCN

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

start_time = time.time()

np.random.seed(32)

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


t = timedelta(seconds= (time.time() - start_time))
print("--- Execution time: %s  ---" % (t))


def evaluate_train():
  for batch in loader:
    batch.to(device)
    pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
    y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

  auroc = AUROC(num_classes=12)
  acc = Accuracy(num_classes=12)
  return auroc(pred,y).item(), acc(pred,y).item()

auroc_t, acc_t = evaluate_train()

print(f"Training ROC-AUC: {auroc_t}, Training accuracy: {acc_t}")


def evaluate_test():
  for batch in test_loader:
    batch.to(device)
    pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
    y = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

  auroc = AUROC(num_classes=12)
  acc = Accuracy(num_classes=12)
  return auroc(pred,y).item(), acc(pred,y).item()

auroc, acc = evaluate_test()

print(f"Test ROC-AUC: {auroc}, Test accuracy: {acc}")


