import os
import torch

device = torch.device("cpu")
print(device)
def parse_hypergraph(filename):
    vertices = {}
    simplices = []
    index_to_labels = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # vertex line
            if line.startswith("v "):
                _, vid, label = line.split()
                try:
                    index = index_to_labels.index(int(label))
                    vertices[int(vid)] = index
                except:
                    index_to_labels.append(int(label))
                    vertices[int(vid)] = index_to_labels.index(int(label))
                # vertices[int(vid)] = label

            else:
                # simplex line
                parts = line.split("-")
                verts = list(map(int, parts[0].split()))
                label = parts[1].strip()
                simplices.append((verts, label))

    return vertices, simplices, index_to_labels


def incidence_matrix(vertices, simplices):
    n = len(vertices)
    m = len(simplices)

    rows = []
    cols = []

    for e_idx, (verts, _) in enumerate(simplices):
        for v in verts:
            rows.append(v)
            cols.append(e_idx)

    indices = torch.tensor([rows, cols])
    values = torch.ones(len(rows))

    H_sparse = torch.sparse_coo_tensor(indices, values, size=(n, m)).to(device)

    return H_sparse


# Usage:
vertices, simplices, index_to_labels = parse_hypergraph("OpenAlex")
H = incidence_matrix(vertices, simplices)
print(index_to_labels)


# In[3]:


len(index_to_labels)


# In[4]:


import numpy as np
from collections import defaultdict

# --------------------
# 1. Load patterns
# --------------------
def load_patterns(pattern_file):
    patterns = []
    with open(pattern_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Time") or line.startswith("Memory") or line.startswith("-"):
                continue
            patterns.append(line)
    return sorted(list(set(patterns)))   # unique patterns

# --------------------
# 2. Load mapping (pattern → node)
# --------------------
def load_pattern_to_nodes(mapping_file):
    pattern_to_nodes = defaultdict(set)

    with open(mapping_file) as f:
        for line in f:
            pattern, node = line.strip().split("\t")
            node = int(node)
            pattern_to_nodes[pattern].add(node)

    return pattern_to_nodes

# --------------------
# 3. Build feature matrix
# --------------------
def build_feature_matrix(patterns, pattern_to_nodes, N):
    P = len(patterns)

    node_to_row = {node: node for node in range(0, N)}
    pattern_to_col = {p: j for j, p in enumerate(patterns)}

    X = np.zeros((N, P), dtype=np.int8)

    for p in patterns:
        col = pattern_to_col[p]
        for node in pattern_to_nodes.get(p, []):
            row = node_to_row[node]
            X[row, col] = 1

    return X, node_to_row, pattern_to_col


# ============================
#       USAGE
# ============================

patterns = load_patterns("OpenAlex_freq_5000_minDim_0_maxSize_5")
pattern_to_nodes = load_pattern_to_nodes("OpenAlex_freq_5000_minDim_0_maxSize_5occMap")

X, node_to_row, pattern_to_col = build_feature_matrix(patterns, pattern_to_nodes, len(vertices))

print("Feature matrix shape:", X.shape)


# In[5]:


def load_vertex_labels(filename):
    """
    Parse vertex labels from file with lines:
    v <node> <label>
    """
    labels = {}  # node_id → label

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                _, node_id, label = line.split()
                node_id = int(node_id)
                label = int(label)   # or str(label) if your labels are strings
                labels[node_id] = index_to_labels.index(label)

    return labels
labels_dict = load_vertex_labels("OpenAlex")


# In[6]:


type(index_to_labels[0])


# In[7]:


N = max(labels_dict.keys()) + 1
y = torch.zeros(N, dtype=torch.long).to(device)

for node, label in labels_dict.items():
    y[node] = label


# In[8]:


print(vertices)


# In[9]:


import random
from collections import defaultdict

def stratified_split(y, train_size=30, val_size=20):
    """
    y: tensor of shape [N] containing class labels
    """
    label_to_nodes = defaultdict(list)

    # Group nodes by their labels
    for node, label in enumerate(y.tolist()):
        label_to_nodes[label].append(node)

    train_idx = []
    val_idx = []
    test_idx = []

    # Stratified sampling for each class
    for label, nodes in label_to_nodes.items():
        nodes = nodes.copy()
        random.shuffle(nodes)

        n = len(nodes)
        t = min(train_size, n)      # in case some classes have fewer samples
        v = min(val_size, n - t)

        train_idx.extend(nodes[:t])
        val_idx.extend(nodes[t:t+v])
        test_idx.extend(nodes[t+v:])  # remaining

    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = stratified_split(y)

print(len(train_idx), len(val_idx), len(test_idx))


# In[10]:


train_mask = torch.zeros(N, dtype=torch.bool).to(device)
val_mask   = torch.zeros(N, dtype=torch.bool).to(device)
test_mask  = torch.zeros(N, dtype=torch.bool).to(device)

train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True


# In[11]:


# def build_edge_features(simplices):
#     edge_labels = []
#     for verts, label in simplices:
#         edge_labels.append(int(label))

#     edge_features = torch.tensor(edge_labels, dtype=torch.long)
#     return edge_features

# edge_features = build_edge_features(simplices)
# print(edge_features)


# ## Section: Hear comes nothing (HMPNN)

# In[12]:


# !pip install topomodelx


# In[13]:


from sklearn.metrics import accuracy_score

from topomodelx.nn.hypergraph.hmpnn import HMPNN

torch.manual_seed(0)

class Network(torch.nn.Module):
    """Network class that initializes the base model and readout layer.

    Base model parameters:
    ----------
    Reqired:
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.

    Optitional:
    **kwargs : dict
        Additional arguments for the base model.

    Readout layer parameters:
    ----------
    out_channels : int
        Dimension of the output features.
    task_level : str
        Level of the task. Either "graph" or "node".
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, task_level="graph", **kwargs
    ):
        super().__init__()

        # Define the model
        self.base_model = HMPNN(
            in_channels=in_channels, hidden_channels=hidden_channels, **kwargs
        )

        # Readout
        self.linear = torch.nn.Linear(hidden_channels, 512)
        self.outputLayer = torch.nn.Linear(512, out_channels)
        self.relu = torch.nn.ReLU(True)
        self.dropout = torch.nn.Dropout(0.3)
        self.out_pool = task_level == "graph"

    def forward(self, x_0, x_1, incidence_1):
        # Base model
        x_0, x_1 = self.base_model(x_0, x_1, incidence_1)

        # Pool over all nodes in the hypergraph
        x = torch.max(x_0, dim=0)[0] if self.out_pool is True else x_0

        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.outputLayer(x)


# In[14]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


# In[15]:


# Base model hyperparameters
in_channels = X.shape[1]
hidden_channels = 256
n_layers = 8

# Readout hyperparameters
out_channels = torch.unique(y).shape[0]
task_level = "graph" if out_channels == 1 else "node"


model = Network(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    n_layers=n_layers,
    task_level=task_level,
).to(device)


# In[16]:


X = torch.tensor(X, dtype=torch.float).to(device)
# edge_features = torch.tensor(edge_features, dtype=torch.float).to(device)


# In[17]:


print(X.shape)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=20,
    min_lr = 1e-6
)

torch.manual_seed(100)
test_interval = 5
num_epochs = 1000


initial_x_1 = torch.zeros((H.shape[1], X.shape[1])).to(device)
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    y_hat = model(X, initial_x_1, H)
    # print(y_hat[train_mask])
    # print(y[train_mask])
    loss = loss_fn(y_hat[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    train_loss = loss.item()
    y_pred = y_hat.argmax(dim=-1)
    train_acc = accuracy_score(y[train_mask].cpu(), y_pred[train_mask].cpu())

    model.eval()
    y_hat = model(X, initial_x_1, H)
    val_loss = loss_fn(y_hat[val_mask], y[val_mask]).item()
    y_pred = y_hat.argmax(dim=-1)
    val_acc = accuracy_score(y[val_mask].cpu(), y_pred[val_mask].cpu())


    # update LR
    scheduler.step(val_acc)

    print(
            f"Epoch: {epoch + 1} train loss: {train_loss:.4f} train acc: {train_acc:.2f} "
            f" val loss: {val_loss:.4f} val acc: {val_acc:.2f}"
        )

    if epoch % test_interval == 0:
        test_loss = loss_fn(y_hat[test_mask], y[test_mask]).item()
        y_pred = y_hat.argmax(dim=-1)
        test_acc = accuracy_score(y[test_mask].cpu(), y_pred[test_mask].cpu())
        print(
            f"Epoch: {epoch + 1} train loss: {train_loss:.4f} train acc: {train_acc:.2f} "
            f" test loss: {test_acc:.4f} test acc: {test_acc:.2f}"
        )


# In[ ]:


y_hat = model(X, initial_x_1, H)
y_pred = y_hat.argmax(dim=-1)
print(y_pred)

print(torch.unique(y_pred))


# In[ ]:


torch.unique(y).shape[0]

