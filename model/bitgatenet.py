import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from utils import SymQuant8bit, log_tensor, LoggerUnit
from .conv2d_q import Conv2d_Q
from .linear_q import Linear_Q
from .dualgatebn_q import DualGateBN_Q


class BitGateNet(nn.Module):
    def __init__(self, num_classes, quantscale=1.0):
        super().__init__()
        self.logger = LoggerUnit("BitGateNet").get_logger()

        self.quantizer = SymQuant8bit(quantscale=quantscale)
        quantizer = self.quantizer  # for shorthand

        self.conv1 = Conv2d_Q(1, 7, kernel_size=3, stride=1, padding=1, quantizer=quantizer)

        self.block1 = DualGateBN_Q(7, 14, 3, 6, 7, quantizer=quantizer)
        self.block2 = DualGateBN_Q(7, 14, 3, 6, 7, quantizer=quantizer)
        self.block3 = DualGateBN_Q(7, 14, 2, 4, 7, quantizer=quantizer)
        self.block4 = DualGateBN_Q(7, 14, 11, 22, 7, quantizer=quantizer)

        self.conv2 = Conv2d_Q(7, 17, kernel_size=3, stride=1, padding=1, quantizer=quantizer)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = Linear_Q(17, num_classes, quantizer=quantizer)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __str__(self):
        return "BitGateNet"

    def forward(self, x):
        x = x.to(self.device)
        log_tensor("00_input", x)
        self.logger.debug(f"Input shape: {x.shape}")

        x = F.relu(self.conv1(x))
        log_tensor("01_conv1_out", x)

        x = self.block1(x)
        log_tensor("02_block1_out", x)

        x = self.block2(x)
        log_tensor("03_block2_out", x)

        x = self.block3(x)
        log_tensor("04_block3_out", x)

        x = self.block4(x)
        log_tensor("05_block4_out", x)

        x = F.relu(self.conv2(x))
        log_tensor("06_conv2_out", x)

        x = self.global_pool(x)
        log_tensor("07_global_pool", x)

        x = x.view(x.size(0), -1)
        log_tensor("08_flattened", x)

        x = self.fc(x)
        log_tensor("09_fc_out", x)

        return x
