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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizerFast, BertModel

# TODO: device

class APRE(nn.Module):
    """Class of rating prediction model APRE
    
    Implemented by PyTorch vTODO.
    
    Note:
        1. Add turn on/off structure
    """

    def __init__(self, args):
        super(APRE, self).__init__()

        # =========================
        #   Settings 
        # =========================
        self.use_exp = not args.disable_explicit
        self.use_imp = not args.disable_implicit
        self.use_cf = not args.disable_cf
        self.nlast_lyr = args.num_last_layers
        self.bert_dim = 768

        # set device
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Whether the model returns all hidden-states.
        self.bert = BertModel.from_pretrained('bert-base-uncased',
            output_hidden_states=True)
        
        # =========================
        #   Aspect related
        # =========================
        # aspect representation
        max_norm = args.aspemb_max_norm if args.aspemb_max_norm > 0 else None
        self.emb_aspect = nn.Embedding(num_embeddings=args.num_aspects,
            max_norm=max_norm)
        
        # =========================
        #   BERT out linears
        # =========================
        self.ubert_out_linear = nn.Linear(bias=True,
            in_features=self.bert_dim*self.nlast_lyr, out_features=self.bert_dim)
        self.ibert_out_linear = nn.Linear(bias=True,
            in_features=self.bert_dim*self.nlast_lyr, out_features=self.bert_dim)
        
        
        # Channel 1 - Explicit

        # Channel 2 - Implicit

        # Channel 3 - CF
        pass


    def forward(self, batch):
        """forward function of APRE

        batch - [Dict] created by easydict. Domains included are `batch_idx`,
            `uid`, `iid`, `unbr`, `inbr`, `urev`, `irev`, `rtg`.
            * urev/irev - (list[EntityReviewAggregation])
            * unbr/inbr - (list[user_id] or list[item_id])
        """
        # combine urev/irev, get split shape
        # feed user revs and item revs
        # split by shapes

        # ==============================
        #   getting BERT encoding
        # ==============================

        # The devices of these things should be GPU!

        # batch.urev, batch.irev list of EntityReviewAggregation
        u_split = [x.get_rev_size() for x in batch.urev]
        i_split = [x.get_rev_size() for x in batch.irev]

        # input_ids
        urevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids']
            for x in batch.urev] #(num_revs*pad_len)
        irevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids']
            for x in batch.irev] #(num_revs*pad_len)
        
        urevs_input_ids = torch.cat(urevs_input_ids, dim=0) # ttl_u_n_rev, padlen
        irevs_input_ids = torch.cat(irevs_input_ids, dim=0) # ttl_i_n_rev, padlen
        
        # attention masks
        urevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask']
            for x in batch.urev] #(num_revs*pad_len)
        irevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask']
            for x in batch.irev] #(num_revs*pad_len)
        
        urevs_attn_mask = torch.cat(urevs_attn_mask, dim=0) # ttl_u_n_rev, padlen
        irevs_attn_mask = torch.cat(irevs_attn_mask, dim=0) # ttl_i_n_rev, padlen

        # user/item review encoding
        # uenc and ienc are both three-item tuples
        # uenc[0] full output seq. dim: (ttl_nrev, pad_len, 768)
        # uenc[1] CLS token output. dim: (ttl_nrev, 768)
        # uenc[2] 13 * (ttl_nrev, pad_len, 768)
        uenc = self.bert(urevs_input_ids, urevs_attn_mask) 
        ienc = self.bert(irevs_input_ids, irevs_attn_mask)

        # get last layers output
        # `ll` is short of `last layers`
        # nlast_lyr* (ttl_nrev, pad_len, 768)
        #   ==> ttl_nrev, pad_len, 768*nlast_lyr
        #   ==> (ttl_nrev, pad_len, 768)
        # TODO: in addition to linear, can also use sum/avg/max pool
        ll_uout = self.ubert_out_linear(
            torch.cat(uenc[2][-self.nlast_lyr:], dim=-1))
        ll_iout = self.ibert_out_linear(
            torch.cat(ienc[2][-self.nlast_lyr:], dim=-1)) # (ttl_nrev, pl, 768)

        # TODO: detail understanding of device!
        # TODO: move all tensor everything onto GPU, esp from dataloader

        # ============================
        #  Process info in 3-channel
        # ============================
        
        if self.use_exp:
            # load sentiment term location within the text
            # before: bs * [nrev [sp-(num_asp*pad_len)]]
            # after : (ttl_nrev) * num_asp * pad_len
            # _rev: EntityReviewAggregation
            urevs_loc = torch.tensor(np.concatenate(
                [x.toarray() for _rev in batch.urev 
                    for x in _rev.get_anno_tkn_revs[1]], axis=0), 
                device=self.device)
            irevs_loc = torch.tensor(np.concatenate(
                [x.toarray() for _rev in batch.irev
                    for x in _rev.get_anno_tkn_revs[1]], axis=0),
                device=self.device)
            



        
        if self.use_imp:
            pass

        if self.use_cf:
            pass




def main():
    # TODO: maybe add some testing
    pass

if __name__ == "__main__":
    main()