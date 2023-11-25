import argparse

import pandas as pd
import numpy as np
import gc
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import ToDense
from torch_geometric.nn import global_mean_pool
from torch import optim
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing 

import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_add_pool
import sys


def create_graphs(filepath):
    '''
    Converts input data to form torch_geometric.data.Data objects
    Args :
        filepath : path to data sets
    Returns :
        list of torch_geometric.data.Data objects
    '''
    graph_labels = pd.read_csv(filepath + 'graph_labels.csv', header=None)
    graph_labels.columns = ['Label']
    num_nodes = pd.read_csv(filepath + 'num_nodes.csv', header=None)
    num_nodes.columns = ['Nodes']
    num_edges = pd.read_csv(filepath + 'num_edges.csv', header=None)
    num_edges.columns = ['Edges']
    graphs = pd.concat([graph_labels, num_nodes, num_edges], axis=1)
    del graph_labels, num_nodes, num_edges
    gc.collect()
    graphs.index = ['G'+str(i+1) for i in range(len(graphs))]
    graphs.replace(to_replace=[np.nan], value=[graphs['Label'].mode()], inplace=True)
    
    node_features = pd.read_csv(filepath + 'node_features.csv', header=None)
    edges = pd.read_csv(filepath + 'edges.csv', header=None)
    edge_features = pd.read_csv(filepath + 'edge_features.csv', header=None)
    
    graph_object = []
    j = 0
    k = 0
    for i in range(len(graphs)):
    # for i in range(1):
        x = torch.as_tensor(node_features[j:j+graphs['Nodes'][i]].to_numpy(), dtype=torch.long)
        j += graphs['Nodes'][i]
        edge_index = torch.as_tensor(edges[k:k+graphs['Edges'][i]].to_numpy().T)
        edge_attr = torch.as_tensor(edge_features[k:k+graphs['Edges'][i]].to_numpy())
        k += graphs['Edges'][i]
        y = float(graphs['Label'][i])
        graph_data = Data(x, edge_index, edge_attr, y)
        graph_object.append(graph_data)
    del node_features, edges, edge_features, graphs
    gc.collect()
    return graph_object

filepath = sys.argv[4]
valid_file_path = sys.argv[6]
model_save_path = sys.argv[2]
graph_objs = create_graphs(filepath)
validation_graphs = create_graphs(valid_file_path)
# draw graph
# g1 = to_networkx(graph_objs[4095])
# nx.draw(g1)
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class GATLayer(MessagePassing):
    def __init__(self, node_in_features, edge_in_features, out_features, heads=2):
        super(GATLayer, self).__init__(aggr='mean')
        self.gat_conv = GATConv(node_in_features, out_features, heads=heads)
        self.edge_lin = nn.Linear(edge_in_features, out_features * heads)

    def forward(self, x, edge_index, edge_attr):
        # Process node features through GAT
        node_features = self.gat_conv(x.float(), edge_index)

        # Process edge features
        edge_features = self.edge_lin(edge_attr.float())

        # Initialize a tensor for aggregated edge features
        edge_features_aggregated = torch.zeros_like(node_features)

        # Aggregate edge features for each node
        for i in range(edge_index.size(1)):
            src_node = edge_index[0, i]
            edge_features_aggregated[src_node] += edge_features[i]

        # Combine node and aggregated edge features
        combined_features = node_features + edge_features_aggregated

        return combined_features

from torch_geometric.nn import global_add_pool

class GAT(nn.Module):
    def __init__(self, num_node_features, num_edge_features, out_features, heads=2):
        super(GAT, self).__init__()
        self.gat_layer1 = GATLayer(num_node_features, num_edge_features, out_features, heads)
        # self.gat_layer2 = GATLayer(out_features * heads, out_features, out_features, heads)
        self.mlp = nn.Sequential(
            nn.Linear(out_features * heads, 32),
        
            
            
            nn.Linear(32, 1),
            
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply GAT layers
        x = self.gat_layer1(x, edge_index, edge_attr)
        # x = self.gat_layer2(x, edge_index, edge_attr)

        # Global mean pooling
        x = global_add_pool(x, data.batch)  # Creates a fixed-size output vector

        # Apply MLP
        out = self.mlp(x)
        return out
import torch
import torch_geometric
from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F



import torch
from torch.nn import MSELoss
from torch_geometric.data import DataLoader
batch_size=32
# Assuming you have a dataset for training
train_loader = DataLoader(graph_objs, batch_size=64, shuffle=False)
torch.manual_seed(42)
model = GAT(9, 3,20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Using MSE Loss
criterion = MSELoss()
regress_loss = []
for epoch in range(70):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        sqrt_loss = torch.sqrt(loss)  # RMSE
        sqrt_loss.backward()
        optimizer.step()
        total_loss += sqrt_loss.item()

    avg_loss = total_loss / batch_size
    regress_loss.append(avg_loss)
    print(f'Epoch {epoch}, Loss: {total_loss / batch_size}')
    

torch.save(model.state_dict(),model_save_path )


def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")


if __name__=="__main__":
    main()
