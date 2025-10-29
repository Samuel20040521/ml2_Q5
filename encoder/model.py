from torch import nn
import numpy as np
import torch
import sys



class Encoder(nn.Module):

    """
    Example model class (not mandatory to use).

    This serves as a simple template to show how a model could be
    initialized and connected with the configuration and C++ backend.
    Students can freely design their own architecture, loss function,
    and training logic based on this structure.
    """

    def __init__(self, loss_device, conf_file, game_type):

        super().__init__()
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
        style_py = _temps.style_py
        style_py.load_config_file(conf_file)

        self.loss_device = loss_device

        

    def loss(self):

    

    def forward(self, inputs):
       

    
