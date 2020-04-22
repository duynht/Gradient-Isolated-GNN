import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class VisionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VisionGNN, self).__init__()

        self.dropout = 0.25
        self.num_layers = 2
        self.hidden = [input_dim, hidden_dim, hidden_dim]
        # self.resnet = models.resnet50(pretrained=True)
        # self.resnet.fc = nn.Identity()

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        for l in range(self.num_layers):
            self.convs.append(self.build_conv_model(
                self.hidden[l], self.hidden[l+1]))
            if (l + 1 < self.num_layers):
                self.lns.append(nn.LayerNorm(self.hidden[l+1]))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))

    def build_conv_model(self, input_dim, hidden_dim):
        # return CustomConv(input_dim, hidden_dim)
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(x.shape)
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            print(type(x), x.shape)
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        # x = pyg_nn.global_mean_pool(x, data.batch)
        x = pyg_nn.global_add_pool(x, data.batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        # Negative log-likelihood
        return F.nll_loss(pred, label)
