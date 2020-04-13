import torchvision
import torch


class ResNetModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.model = self._create_full_model(opt)

        print(self.model)

    def _create_full_model(self, opt):

        # encoder = torch.nn.Sequential()
        # encoder = torchvision.models.resnet50(pretrained=True)
        # for param in encoder.parameters():
        #     param.requires_grad = False

        # encoder.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(
        #     7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # encoder.conv1.requires_grad = True

        # encoder.fc = torch.nn.Linear(
        #     in_features=2048,
        #     out_features=256
        # )
        # encoder.fc.requires_grad = True

        return torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        return self.model(x)
