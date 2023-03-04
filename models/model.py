import torch.nn as nn
from util import Interpolate
import torchvision.models as models


class NetConv2(nn.Module):
    def __init__(self):
        super(NetConv2, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        # Freeze Encoder parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.features = self.pretrained_model.features[:6]

        self.net_conv_2 = nn.Sequential(
            nn.Conv2d(192, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 256, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = self.net_conv_2(x)
        x = Interpolate(x)
        return x


class NetConv5(nn.Module):
    def __init__(self):
        super(NetConv5, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        # Freeze Encoder parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.net_conv_5 = nn.Sequential(
            ############################   Conv Layers   ###############################
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ############################   deConv Layers    #############################
            nn.ConvTranspose2d(256, 256, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = self.net_conv_5(x)
        x = Interpolate(x)
        return x


class NetFC6(nn.Module):
    def __init__(self):
        super(NetFC6, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        # Freeze Encoder parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.classifier = self.pretrained_model.classifier[:4]

        self.net_FC6_before_reshape = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.net_FC6_after_reshape = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.net_FC6_before_reshape(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.net_FC6_after_reshape(x)
        x = Interpolate(x)
        return x


class NetFC8(nn.Module):
    def __init__(self):

        super(NetFC8, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        # Freeze Encoder parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.net_FC8_before_reshape = nn.Sequential(
            nn.Linear(in_features=1000, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.net_FC8_after_reshape = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.net_FC8_before_reshape(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.net_FC8_after_reshape(x)
        x = Interpolate(x)
        return x
