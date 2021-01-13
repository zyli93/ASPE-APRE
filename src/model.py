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

from transformers import BertModel
# from transformers import DistilBertModel

from utils import check_memory

# TODO: verify if uid/iid are int

class APRE(nn.Module):
    """Class of rating prediction model APRE
    
    Implemented by PyTorch vTODO.
    
    Note:
        1. Add turn on/off structure
    """

    def __init__(self, args):
        super(APRE, self).__init__()

        # TODO: ADD initialization to all Linear!
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

        # =========================
        #   Settings 
        # =========================
        self.use_exp = not args.disable_explicit
        self.use_imp = not args.disable_implicit
        self.use_cf = not args.disable_cf
        self.nlast_lyr = args.num_last_layers
        self.num_asp = args.num_aspects
        self.feat_dim = args.feat_dim # TODO: add use to feat_dim
        self.bert_dim = 256
        # self.bert_dim = 768
        self.ex_temp = args.ex_attn_temp
        self.im_temp = args.im_attn_temp
        self.pad_len = args.padded_length

        # Whether the model returns all hidden-states.
        with torch.no_grad():
            self.bert = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4', output_hidden_states=True)
            # self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

        # =========================
        #   Aspect related
        # =========================
        # aspect representation
        max_norm = args.aspemb_max_norm if args.aspemb_max_norm > 0 else None
        # self.emb_aspect = nn.Embedding(num_embeddings=args.num_aspects, max_norm=max_norm, embedding_dim=) # TODO
        self.emb_aspect = nn.Parameter(torch.randn(self.num_asp, self.feat_dim))

        # TODO: remove below if not used
        # review-wise att
        # self.urev_dim_attn = nn.Linear(in_features=self.bert_dim*2, out_features=1, bias=False)
        # self.irev_dim_attn = nn.Linear(in_features=self.bert_dim*2, out_features=1, bias=False)
        
        # =========================
        #   BERT out linears
        # =========================
        self.ubert_out_linear = nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.feat_dim)
        self.ibert_out_linear = nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.feat_dim)
        
        # ============================================
        # Explicit - Review-wise Attention Linear Layer
        # ============================================
        self.ex_urev_attn_linear = nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1)
        self.ex_irev_attn_linear = nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1)

        # ============================================
        # Implicit - Review-wise Attention Linear Layer
        # ============================================
        self.im_urev_attn_linear = nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1)
        self.im_irev_attn_linear = nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1)

        self.im_u_cls_linear = nn.Linear(bias=False, in_features=self.bert_dim, out_features=self.feat_dim)
        self.im_i_cls_linear = nn.Linear(bias=False, in_features=self.bert_dim, out_features=self.feat_dim)

        # [Removed from the final design] Channel 3 - CF
        
        # ============================
        #  Final MLP Layers
        # ============================
        self.ex_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.feat_dim*2, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Linear(bias=False, in_features=self.feat_dim, out_features=1))
        
        # it's different that input shape is (..., feat_dim*4)
        self.im_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.feat_dim*4, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Linear(bias=False, in_features=self.feat_dim, out_features=1))

        # bias term and weight gamma
        self.b_u = nn.Embedding(num_embeddings=args.num_user, embedding_dim=1)
        self.b_t = nn.Embedding(num_embeddings=args.num_item, embedding_dim=1)

        # gamma dim: (num_asp, 1)
        self.gamma = nn.Parameter(torch.randn(args.num_aspects))

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
        # u_split = [x.get_rev_size() for x in batch.urev]
        # i_split = [x.get_rev_size() for x in batch.irev]
        u_split, i_split = batch['u_split'], batch['i_split']

        # contextualized encoder (bert) input_ids
        # urevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in batch.urev] # list of (num_u_revs*pad_len)
        # irevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in batch.irev] # list of (num_u_revs*pad_len)
        # urevs_input_ids = torch.cat(urevs_input_ids, dim=0) # ttl_u_n_rev, padlen
        # irevs_input_ids = torch.cat(irevs_input_ids, dim=0) # ttl_i_n_rev, padlen
        urevs_input_ids, irevs_input_ids = batch['urevs_input_ids'], batch['irevs_input_ids']
        
        # contextualized encoder (bert) attention masks
        # urevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in batch.urev] #(num_u_revs*pad_len)
        # irevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in batch.irev] #(num_u_revs*pad_len)
        # urevs_attn_mask = torch.cat(urevs_attn_mask, dim=0) # ttl_u_n_rev, padlen
        # irevs_attn_mask = torch.cat(irevs_attn_mask, dim=0) # ttl_i_n_rev, padlen
        urevs_attn_mask, irevs_attn_mask = batch['urevs_attn_mask'], batch['irevs_attn_mask']

        # user/item review encoding by BERT
        '''u_bert_out and i_bert_out are both three-item tuples
          u_bert_out[0] full output seq. dim: (ttl_nrev, pad_len, 768)
          u_bert_out[1] CLS token output. dim: (ttl_nrev, 768)
          u_bert_out[2] 13 * (ttl_nrev, pad_len, 768)
          (same as i_bert_out)'''
        u_bert_out = self.bert(urevs_input_ids, urevs_attn_mask) 

        i_bert_out = self.bert(irevs_input_ids, irevs_attn_mask)

        # print("u/t batch size", urevs_input_ids.shape, irevs_input_ids.shape)

        # get last layers output, (`ll` is short of `last layers`)
        # nlast_lyr * (ttl_nrev, pad_len, 768) ==> ttl_nrev, pad_len, 768*nlast_lyr
        #   ==> (ttl_nrev, pad_len, feat_dim)  (`pl` as pad_len)
        # TODO: in addition to linear, can also use sum/avg/max pool
        # TODO: try to add this dense layer's activation, and use non-linear matmul
        urev_lang_enc = self.ubert_out_linear(torch.cat(u_bert_out[2][-self.nlast_lyr:], dim=-1))
        irev_lang_enc = self.ibert_out_linear(torch.cat(i_bert_out[2][-self.nlast_lyr:], dim=-1)) # (ttl_nrev, pl, 768)

        # TODO: move all tensor everything onto GPU, esp from dataloader

        # ===================================
        #  Process info in Ex and Im Channels
        # ===================================
        
        # Handle Explicit information
        if self.use_exp:
            # Before: bs * [nrev [sp_mat(num_asp*pad_len)]]; After: (ttl_nrev) * num_asp * pad_len
            # _rev: EntityReviewAggregation
            # NOTE: using `torch.as_tensor()` to avoid data copy
            # urevs_loc = torch.as_tensor(np.concatenate(
            #     [x.toarray() for _rev in batch.urev for x in _rev.get_anno_tkn_revs()[1]], axis=0), dtype=torch.float)
            # irevs_loc = torch.as_tensor(np.concatenate(
            #     [x.toarray() for _rev in batch.irev for x in _rev.get_anno_tkn_revs()[1]], axis=0), dtype=torch.float)
            urevs_loc, irevs_loc = batch['urevs_loc'], batch['irevs_loc']

            # user/item aspect representation
            # (ttl_nrev, num_asp, pad_len) [matmul] (ttl_nrev, pad_len, feat_dim)
            # generates (ttl_nrev, num_asp, feat_dim)
            # print("urevs_loc.shape", urevs_loc.shape)
            # print("urev_lang_enc.shape", urev_lang_enc.shape)
            uasp_repr = torch.matmul(urevs_loc.view(-1, self.num_asp, self.pad_len), urev_lang_enc)
            iasp_repr = torch.matmul(irevs_loc.view(-1, self.num_asp, self.pad_len), irev_lang_enc)

            # print("uasp_repr shape")
            # print(uasp_repr.shape)

            # print("usplit")
            # print(u_split)

            # [list] bs * tuple of (nrev,num_asp,feat_dim)
            uasp_repr_spl = torch.split(uasp_repr, u_split, dim=0)
            iasp_repr_spl = torch.split(iasp_repr, i_split, dim=0)
            
            # get attention aggregation
            uasp_repr_agg, iasp_repr_agg = [], []
            for i in range(len(u_split)):
                # NOTE: method2 without concatenation
                num_u_rev = u_split[i]
                num_i_rev = i_split[i]

                # duplicate dim[0] of emb_aspect by num_u_rev times
                # ex_u_emb_aspect = self.emb_aspect.unsqueeze_(0).expand(num_u_rev)
                ex_u_emb_aspect = torch.unsqueeze(self.emb_aspect, 0).expand(num_u_rev, -1, -1)
                # print(ex_u_emb_aspect.shape)
                # print(uasp_repr_spl[i].shape)

                # After cat => (nrev, num_asp, 2*feat_dim); After linear => nrev*num_asp*1
                # [OUTPUT] urev_attn_w & irev_attn_w can be output!
                urev_attn_w = F.softmax(dim=0, input=F.tanh(
                        self.ex_urev_attn_linear(torch.cat((uasp_repr_spl[i], ex_u_emb_aspect), dim=2))) / self.ex_temp)

                # (nrev, num_asp, 1) mul (nrev, num_asp, feat_dim)
                # After mul => (nrev, num_asp, feat_dim), sum => (num_asp, feat_dim)
                uasp_repr_revagg = torch.sum(torch.mul(urev_attn_w, uasp_repr_spl[i]), dim=0, keepdim=False)
                
                # print("iasp_repr_spl[i].shape")
                # print(iasp_repr_spl[i].shape)
                ex_i_emb_aspect = torch.unsqueeze(self.emb_aspect, 0).expand(num_i_rev, -1, -1)
                irev_attn_w = F.softmax(dim=0, input=F.tanh(
                        self.ex_irev_attn_linear(torch.cat((iasp_repr_spl[i], ex_i_emb_aspect), dim=2))) / self.ex_temp)
                iasp_repr_revagg = torch.sum(torch.mul(irev_attn_w, iasp_repr_spl[i]), dim=0, keepdim=False)
                
                uasp_repr_agg.append(uasp_repr_revagg)
                iasp_repr_agg.append(iasp_repr_revagg)
            
            # bs*num_asp* feat_dim
            u_expl_repr = torch.stack(uasp_repr_agg, dim=0)
            i_expl_repr = torch.stack(iasp_repr_agg, dim=0)

        
        # Handle Implicit information 
        if self.use_imp:
            # urev_lang_enc/irev_lang_enc (ttl_nrev, pad_len, 768)
            
            # fetch [CLS] token representation, (ttl_nrev, bert_dim)
            # urevs_cls_repr, irevs_cls_repr = u_bert_out[1], i_bert_out[1]

            # [new] fetch [CLS] token representation, (ttl_nrew, feat_dim)
            urevs_cls_repr = self.im_u_cls_linear(u_bert_out[1])
            irevs_cls_repr = self.im_i_cls_linear(i_bert_out[1])
            
            # After sum => (ttl_nrev, 768); After div => (ttl_nrev, 768)
            urevs_avgpool_repr = torch.div(torch.sum(urev_lang_enc, dim=1), torch.sum(urev_lang_enc!=0, dim=1))
            irevs_avgpool_repr = torch.div(torch.sum(irev_lang_enc, dim=1), torch.sum(irev_lang_enc!=0, dim=1))
            
            # concat=> ttl_nrev, feat_dim + 768
            uconcat_repr = torch.cat([urevs_cls_repr, urevs_avgpool_repr], dim=-1)
            iconcat_repr = torch.cat([irevs_cls_repr, irevs_avgpool_repr], dim=-1)

            uimpl_repr_spl = torch.split(uconcat_repr, u_split)
            iimpl_repr_spl = torch.split(iconcat_repr, i_split)

            uimpl_repr_agg, iimpl_repr_agg = [], []
            for i in range(len(u_split)):
                # Before: (nrev, feat_dim)
                # linear => nrev*1, tanh, softmax (no change on size)
                uimpl_rev_w = F.softmax(dim=0, input=F.tanh(self.im_urev_attn_linear(uimpl_repr_spl[i])))
                iimpl_rev_w = F.softmax(dim=0, input=F.tanh(self.im_irev_attn_linear(iimpl_repr_spl[i])))

                # print("uimpl_rev_w.shape")
                # print(uimpl_rev_w.shape)

                # print("uimpl_repr_spl[i].shape")
                # print(uimpl_repr_spl[i].shape)
                
                uimpl_rev_attn_agg = torch.sum(torch.mul(uimpl_repr_spl[i], uimpl_rev_w), dim=0)
                iimpl_rev_attn_agg = torch.sum(torch.mul(iimpl_repr_spl[i], iimpl_rev_w), dim=0)
                
                uimpl_repr_agg.append(uimpl_rev_attn_agg)
                iimpl_repr_agg.append(iimpl_rev_attn_agg)
            
            # bs, feat_dim
            u_impl_repr = torch.stack(uimpl_repr_agg, dim=0)
            i_impl_repr = torch.stack(iimpl_repr_agg, dim=0)


        if self.use_cf:
            pass
            
        # ==========================
        #   Merge Ex & Im channels
        # ==========================
        # create tensor for user ID and item ID
        batch_uid, batch_iid = batch['uid'], batch['iid']

        # first three terms: (bs, 1), (bs, 1), (bs, 1)
        pred = self.b_u(batch_uid) + self.b_t(batch_iid) + self.im_mlp(torch.cat(
            (u_impl_repr, i_impl_repr), dim=-1))
        pred = pred.squeeze()  # (bs)

        # (bs, num_asp, 2*feat_dim) => MLP => (bs, num_asp, 1) => squeeze => (bs, num_asp)
        # (bs, num_asp) matmul (num_asp, 1) -> (bs, 1) -> (bs)
        pred += torch.matmul(
            self.ex_mlp(torch.cat((u_expl_repr, i_expl_repr), dim=-1)).squeeze(), 
            torch.unsqueeze(self.gamma, 1)).squeeze()

        return pred