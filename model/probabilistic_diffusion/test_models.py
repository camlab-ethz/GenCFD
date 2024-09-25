import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import dataloader

from model.building_blocks.unets.unets import UNet

class TestDenoisingModel(unittest.TestCase):
    def test_denoiser(self):
        test_cases = [
            ((64, 64), "CIRCULAR", (2, 2, 2), False),
        ]