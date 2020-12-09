'''
    The file of the model APRE

    File name: model.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
    PyTorch Version: TODO
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizerFast, BertModel


class APRE(nn.Module):
    """Class of rating prediction model APRE
    
    Implemented by PyTorch vTODO.
    
    Note:
        1. Add turn on/off structure
    """

    def __init__(self):
        super(APRE, self).__init__()

        # Channel 1 - Explicit

        # Channel 2 - Implicit

        # Channel 3 - CF
        pass


    def forward(x):
        """forward function of APRE"""
        pass




def main():
    # TODO: maybe add some testing
    pass

if __name__ == "__main__":
    main()