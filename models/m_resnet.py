import torch.nn.functional as F
from torch import nn
import torch
from models.resnet import *

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder,self).__init__()

        self.restored = False

        #output [batchsize, 64, 56, 56]
        self.encoder = resnet34(pretrained=True, in_feature=3, mid_out=True)

        for p in self.encoder.parameters():
            p.requires_grad=True

    def forward(self, input):
        conv_out = self.encoder(input)
        feat = conv_out.view(-1, 512)
        return feat

class ResnetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(ResnetClassifier, self).__init__()
        self.restored = False
        self.fc = nn.Linear(512, 2)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        # out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc(feat)
        return out