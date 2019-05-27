import torch.nn as nn
import torch
from models.resnet import *
import numpy as np

class CloudImageEncoder(nn.Module):
    def __init__(self):
        super(CloudImageEncoder, self).__init__()
        self.restored = False
        self.cloudImage_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.Tanh(),
        )
        self.seq = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 200),
            nn.Tanh()
        )

    def forward(self, input):
        conv_out = self.cloudImage_encoder(input)
        flat_conv = conv_out.view(-1, 512)
        seq = self.seq(flat_conv)
        encoder_out = seq.view(-1, 200, 1, 1, 1)
        return encoder_out

class CloudEncoderDiscriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(CloudEncoderDiscriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.Sigmoid()
            # nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class CloudImageEncoderResNet(nn.Module):
    def __init__(self):
        super(CloudImageEncoderResNet, self).__init__()
        self.restored = False
        self.cloudImage_encoder = resnet18(pretrained=True, in_feature=3, mid_out=True)
        for p in self.cloudImage_encoder.parameters():
            p.requires_grad = True
        self.seq = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 200),
            nn.Tanh()
        )

    def forward(self, input):
        conv_out = self.cloudImage_encoder(input)
        flat_conv = conv_out.view(-1, 512)
        seq = self.seq(flat_conv)
        encoder_out = seq.view(-1, 200, 1, 1, 1)
        return encoder_out


class CloudDiscriminator(nn.Module):
    def __init__(self):
        super(CloudDiscriminator, self).__init__()
        self.restored = False
        self.discriminator = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.discriminator(input)
        return output


class Cloud3DEncoder(nn.Module):
    def __init__(self):
        super(Cloud3DEncoder, self).__init__()
        self.restored = False
        self.cloud3D_encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, kernel_size=4, stride=1),
            nn.BatchNorm3d(512),
            nn.Tanh()
        )
        self.seq = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 200),
            nn.Tanh()
        )

    def forward(self, input):
        conv_out = self.cloud3D_encoder(input)
        flat_conv = conv_out.view(-1, 512)
        seq = self.seq(flat_conv)
        encoder_out = seq.view(-1, 200, 1, 1, 1)
        return encoder_out

class Cloud3DDecoder(nn.Module):
    def __init__(self):
        super(Cloud3DDecoder, self).__init__()
        self.restored = False
        self.cloud3D_decoder = nn.Sequential(
            nn.ConvTranspose3d(200, 512, kernel_size=4, stride=1),
            # nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
        )

        self.cloud3D_decoder_detail = nn.Sequential(
            nn.ConvTranspose3d(200, 512, kernel_size=4, stride=1),
            # nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
        )

        self.tanh = nn.Tanh()
        self.maxPool = nn.MaxPool3d((1, 64, 1), stride=(1, 1, 1))


    def forward(self, input, cloud_mean):
        cloud_out = self.tanh(self.cloud3D_decoder(input) + self.cloud3D_decoder_detail(input))
        cloud_mask = self.maxPool(cloud_out + cloud_mean).squeeze(3)
        return cloud_out, cloud_mask

if __name__ == '__main__':
    # model = CloudImageEncoderResNet()
    # input = torch.rand(8, 3, 224, 224)
    # out = model(input)
    # decoder = Cloud3DDecoder()
    # cloud_mean = torch.rand(8, 1, 64, 64, 64)
    # cloud_out, cloud_mask = decoder(out, cloud_mean)
    # print(out.size())
    # print(cloud_out.size())
    model = CloudEncoderDiscriminator(input_dims=200, output_dims=1, hidden_dims=100)
    input = torch.rand(16, 200)
    out = model(input)
    print(out.size())