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
        self.num_asp = args.num_aspects
        self.bert_dim = 768

        # Whether the model returns all hidden-states.
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        # =========================
        #   Aspect related
        # =========================
        # aspect representation
        max_norm = args.aspemb_max_norm if args.aspemb_max_norm > 0 else None
        self.emb_aspect = nn.Embedding(num_embeddings=args.num_aspects, max_norm=max_norm)
        
        # aspect embedding num_asp * 768
        self.emb_asp = nn.Parameters(torch.randn(self.num_asp, self.bert_dim))

        # review-wise att
        self.urev_dim_attn = nn.Linear(in_features=self.bert_dim*2, out_features=1, bias=False)
        self.irev_dim_attn = nn.Linear(in_features=self.bert_dim*2, out_features=1, bias=False)
        
        # =========================
        #   BERT out linears
        # =========================
        self.ubert_out_linear = nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.bert_dim)
        self.ibert_out_linear = nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.bert_dim)
        
        
        # ==========================
        # For Explicit
        # ==========================
        self.urev_attn_linear = nn.Linear(bias=False, in_features=self.bert_dim, out_features=1)
        self.irev_attn_linear = nn.Linear(bias=False, in_features=self.bert_dim, out_features=1)

        # Channel 2 - Implicit
        self.urev_attn_linear_impl = nn.Linear(bias=False, in_features=self.bert_dim, out_features=1)
        self.irev_attn_linear_impl = nn.Linear(bias=False, in_features=self.bert_dim, out_features=1)

        # Channel 3 - CF
        # Pass for now.
        
        # ============================
        #  Final
        # ============================
        self.ex_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.bert_dim*2, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(bias=False, in_features=self.mid_dim, out_features=1))
        
        self.im_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.bert_dim*2, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(bias=False, in_features=self.mid_dim, out_features=1))

        # bias term and weight gamma
        self.b_u = nn.Embedding(num_embeddings=args.num_user, embedding_dim=1)
        self.b_t = nn.Embedding(num_embeddings=args.num_item, embedding_dim=1)
        # (num_asp, 1)
        self.gamma = nn.Parameters(torch.randn(args.num_aspects)).unsqueeze_(1)  

        # Loss Func
        self.loss_func = nn.MSELoss()


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
        urevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in batch.urev] #(num_revs*pad_len)
        irevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in batch.irev] #(num_revs*pad_len)
        
        urevs_input_ids = torch.cat(urevs_input_ids, dim=0) # ttl_u_n_rev, padlen
        irevs_input_ids = torch.cat(irevs_input_ids, dim=0) # ttl_i_n_rev, padlen
        
        # attention masks
        urevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in batch.urev] #(num_revs*pad_len)
        irevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in batch.irev] #(num_revs*pad_len)
        
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
        # nlast_lyr* (ttl_nrev, pad_len, 768) ==> ttl_nrev, pad_len, 768*nlast_lyr
        #   ==> (ttl_nrev, pad_len, 768)
        # TODO: in addition to linear, can also use sum/avg/max pool
        # TODO: try to remove this dense layer's activation, and use linear matmul
        ull_out = self.ubert_out_linear(torch.cat(uenc[2][-self.nlast_lyr:], dim=-1))
        ill_out = self.ibert_out_linear(torch.cat(ienc[2][-self.nlast_lyr:], dim=-1)) # (ttl_nrev, pl, 768)

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
                [x.toarray() for _rev in batch.urev for x in _rev.get_anno_tkn_revs()[1]], axis=0))
            irevs_loc = torch.tensor(np.concatenate(
                [x.toarray() for _rev in batch.irev for x in _rev.get_anno_tkn_revs()[1]], axis=0))

            # user/item representation
            uasp_repr = torch.matmul(urevs_loc, ull_out)  # ttl_nrev, num_asp, 768
            iasp_repr = torch.matmul(irevs_loc, ill_out)

            # bs tuple of (nrev*num_asp*768)
            uasp_repr_spl = torch.split(uasp_repr, u_split, dim=0)
            iasp_repr_spl = torch.split(iasp_repr, i_split, dim=0)
            
            # get attention agg
            uasp_repr_agg, iasp_repr_agg = [], []
            for i in range(len(u_split)):
                # NOTE: method2 without concatenation
                num_rev = u_split[i]
                exp_emb_aspect = self.emb_aspect.unsqueeze_(0).expand(num_rev)

                # cat => nrev*num_asp*(768+768)
                # linear => nrev*num_asp*1
                urev_attn_w = F.softmax(dim=0, input=F.tanh(
                        self.urev_attn_linear(torch.cat((uasp_repr_spl[i], exp_emb_aspect), dim=2))))
                # mul => nrev*num_asp*768, sum => num_asp*768
                uasp_repr_revagg = torch.sum(torch.mul(urev_attn_w, uasp_repr_spl[i]), dim=0, keepdim=False)
                
                irev_attn_w = F.softmax(dim=0, input=F.tanh(
                        self.irev_attn_linear(torch.cat((iasp_repr_spl[i], exp_emb_aspect), dim=2))))
                iasp_repr_revagg = torch.sum(torch.mul(irev_attn_w, iasp_repr_spl[i]), dim=0, keepdim=False)
                
                uasp_repr_agg.append(uasp_repr_revagg)
                iasp_repr_agg.append(iasp_repr_revagg)
            
            # bs*num_asp* 768
            u_expl_repr = torch.stack(uasp_repr_agg, dim=0)
            i_expl_repr = torch.stack(iasp_repr_agg, dim=0)

            # TODO

        
        if self.use_imp:
            # handling ull_out/ill_out (ttl_nrev, pad_len, 768)
            
            # [CLS] rep, (ttl_nrev, 768)
            urevs_cls_repr, irevs_cls_repr = uenc[1], ienc[1]
            
            # avgpooling without sum=> ttl_nrev, 768
            # div => (ttl_nrev, 768)
            urevs_avgpool_repr = torch.div(torch.sum(ull_out, dim=1), torch.sum(ull_out!=0, dim=1))
            irevs_avgpool_repr = torch.div(torch.sum(ill_out, dim=1), torch.sum(ill_out!=0, dim=1))
            
            # concat=> ttl_nrev, 768*2
            uimpl = torch.cat([urevs_cls_repr, urevs_avgpool_repr], dim=-1)
            iimpl = torch.cat([irevs_cls_repr, irevs_avgpool_repr], dim=-1)

            uimpl_spl = torch.split(uimpl, u_split)
            iimpl_spl = torch.split(iimpl, i_split)

            uimpl_repr_agg, iimpl_repr_agg = [], []
            for i in range(len(u_split)):
                # nrev*768 
                # linear => nrev*1, tanh, softmax (no change on size)
                uimpl_rev_w = F.softmax(dim=0, input=F.tanh(self.urev_attn_linear_impl(uimpl_spl[i])))
                iimpl_rev_w = F.softmax(dim=0, imput=F.tanh(self.irev_attn_linear_impl(iimpl_spl[i])))
                
                uimpl_rev_attn_agg = torch.sum(torch.mul(uimpl_spl[i], uimpl_rev_w), dim=0)
                iimpl_rev_attn_agg = torch.sum(torch.mul(iimpl_spl[i], iimpl_rev_w), dim=0)
                
                uimpl_repr_agg.append(uimpl_rev_attn_agg)
                iimpl_repr_agg.append(iimpl_rev_attn_agg)
            
            # bs*768
            u_impl_repr = torch.stack(uimpl_repr_agg, dim=0)
            i_impl_repr = torch.stack(iimpl_repr_agg, dim=0)


        if self.use_cf:
            pass
            # TODO: verify if uid/iid are int
            
        # ======================
        #   Merge three channel
        # ======================
        # first three terms: (bs, 1), (bs, 1), (bs, 1)
        # create tensor for them
        batch_uid, batch_iid = torch.from_numpy(batch.users), torch.from_numpy(batch.items)
        pred = self.b_u[batch_uid] + self.b_t[batch_iid] + self.im_mlp(torch.cat(u_impl_repr, i_impl_repr))
        pred = pred.squeeze()  # (bs)

        # (bs, num_asp, 2*768) -> (bs, num_asp, 1) -> (bs, num_asp)
        # (bs, num_asp)*(num_asp, 1) -> (bs, 1) -> (bs)
        pred += torch.matmul(
            self.ex_mlp(torch.cat(u_expl_repr, i_expl_repr)).squeeze(), self.gamma).squeeze()

        return pred


def main():
    # TODO: maybe add some testing
    pass

if __name__ == "__main__":
    main()