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

    def __init__(self, args):
        super(APRE, self).__init__()

        # settings 
        self.use_exp = not args.disable_explicit
        self.use_imp = not args.disable_implicit
        self.use_cf = not args.disable_cf

        # Whether the model returns all hidden-states.
        self.bert = BertModel.from_pretrained('bert-base-uncased',
            output_hidden_states=True)
        
        # aspect representation
        max_norm = args.aspemb_max_norm if args.aspemb_max_norm > 0 else None
        self.emb_aspect = nn.Embedding(num_embeddings=args.num_aspects,
            max_norm=max_norm)
        
        # Channel 1 - Explicit

        # Channel 2 - Implicit

        # Channel 3 - CF
        pass


    def forward(batch):
        """forward function of APRE

        batch - [Dict] created by easydict. Domains included are `batch_idx`,
            `uid`, `iid`, `unbr`, `inbr`, `urev`, `irev`, `rtg`.
            * urev/irev - (list[EntityReviewAggregation])
            * unbr/inbr - (list[user_id] or list[item_id])
        """
        # combine urev/irev, get split shape
        # feed user revs and item revs
        # split by shapes

        if self.use_exp:
            pass
        
        if self.use_imp:
            pass

        if self.use_cf:
            pass




def main():
    # TODO: maybe add some testing
    pass

if __name__ == "__main__":
    main()