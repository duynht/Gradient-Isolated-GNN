import torch
import torch.nn as nn


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=200, hidden_nodes=0, avg_pooling_kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d(
            (avg_pooling_kernel_size, avg_pooling_kernel_size), stride=0, padding=0)
        self.model = nn.Sequential()

        if hidden_nodes > 0:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=0.5))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )

        else:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
            )

        print(self.model)

    def forward(self, x, *args):
        x = self.avg_pool(x).squeeze()
        x = x.view(x.size(0), -1)
        x = self.model(x).squeeze()
        return x


class FusionClassificationModel(torch.nn.Module):
    def __init__(self, num_classes=200, vision_output_dim=1, final_vision_vector_size=1024, word_vector_size=3300):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((vision_output_dim, vision_output_dim))
        self.model = nn.Sequential()
        
        self.model.add_module(
            "layer1", nn.Linear(final_vision_vector_size + word_vector_size, num_classes, bias=True)
        )

        print(self.model)

    def forward(self, inputs, *args):
        print("==")
        x, word_vectors = inputs['img'], inputs['desc']
        print(x.shape)
        x = self.avg_pool(x).squeeze()
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        word_vectors = word_vectors.view(word_vectors.size(0), -1)
        print(word_vectors.shape)
        x = torch.cat((x, word_vectors.float()), dim=1)
        print(x.shape)
        x = self.model(x).squeeze()
        return x
