'''
    The file of the model APRE

    File name: model.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
    PyTorch Version: TODO

    # some once useful notes:
            # NOTE: using `torch.as_tensor()` to avoid data copy
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

from utils import check_memory

class APRE(nn.Module):
    """Class of rating prediction model APRE
    
    Implemented by PyTorch vTODO.
    
    Note:
        1. Add turn on/off structure
    """

    def __init__(self, args):
        super(APRE, self).__init__()

        # TODO: add save input data after run once!
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

        # =========================
        #   Settings 
        # =========================
        self.use_exp = not args.disable_explicit
        self.use_imp = not args.disable_implicit
        self.use_cf = not args.disable_cf
        # self.nlast_lyr = args.num_last_layers
        self.nlast_lyr = 1
        self.num_asp = args.num_aspects
        self.feat_dim = args.feat_dim 
        self.bert_dim = 256
        # self.bert_dim = 768
        self.pad_len = args.padded_length
        self.out_channel = args.cnn_out_channel
        self.dropout_rate = args.dropout
        self.transf_wordemb_func = args.transf_wordemb_func

        self.im_dim = 3 * self.feat_dim + self.out_channel
        print("model im_dim setting {}".format(self.im_dim))

        # Aspect Representation
        self.u_emb_aspect = nn.Parameter(torch.randn(self.num_asp, self.feat_dim))  # (num_asp, feat_dim)
        self.i_emb_aspect = nn.Parameter(torch.randn(self.num_asp, self.feat_dim))  # (num_asp, feat_dim)

        # =========================
        #   BERT output linears
        # =========================
        if self.transf_wordemb_func == "leakyrelu":
            emb_transf = nn.LeakyReLU
        elif self.transf_wordemb_func == "relu":
            emb_transf = nn.ReLU
        elif self.transf_wordemb_func == "tanh":
            emb_transf = nn.Tanh
        else:
            emb_transf = nn.Identity

        # TODO: [tuning] is leaky a good option?
        self.ubert_out_linear = nn.Sequential(
            nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.feat_dim),
            emb_transf())
        self.ibert_out_linear = nn.Sequential(
            nn.Linear(bias=True, in_features=self.bert_dim*self.nlast_lyr, out_features=self.feat_dim),
            emb_transf())
        
        # ============================================
        # Explicit - Review-wise Attention Linear Layer
        # ============================================
        self.ex_urev_attn_lin_tanh = nn.Sequential(
            nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1),
            nn.Tanh())

        self.ex_irev_attn_lin_tanh = nn.Sequential(
            nn.Linear(bias=False, in_features=2*self.feat_dim, out_features=1),
            nn.Tanh())

        # ============================================
        # Implicit - Review-wise Attention Linear Layer
        # ============================================
        self.im_urev_attn_lin_tanh = nn.Sequential(
            nn.Linear(bias=False, in_features=self.im_dim, out_features=1),
            nn.Tanh())
        self.im_irev_attn_lin_tanh = nn.Sequential(
            nn.Linear(bias=False, in_features=self.im_dim, out_features=1),
            nn.Tanh())

        self.im_u_cls_linear = nn.Sequential(
            nn.Linear(bias=False, in_features=self.bert_dim, out_features=self.feat_dim),
            emb_transf())
        self.im_i_cls_linear = nn.Sequential(
            nn.Linear(bias=False, in_features=self.bert_dim, out_features=self.feat_dim),
            emb_transf())

        self.im_u_cnn = nn.Conv2d(in_channels=1, out_channels=args.cnn_out_channel, kernel_size=(args.im_kernel_size, self.feat_dim))
        self.im_i_cnn = nn.Conv2d(in_channels=1, out_channels=args.cnn_out_channel, kernel_size=(args.im_kernel_size, self.feat_dim))

        # ============================
        #  Final MLP Layers
        # ============================
        self.ex_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.feat_dim*2, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(bias=False, in_features=self.feat_dim, out_features=1))
        
        # it's different that input shape is (..., feat_dim*4)
        self.im_mlp = nn.Sequential(
            nn.Linear(bias=True, in_features=self.im_dim*2, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(bias=False, in_features=self.feat_dim, out_features=1))

        # bias term and weight gamma
        self.b_u = nn.Embedding(num_embeddings=args.num_user, embedding_dim=1)
        self.b_t = nn.Embedding(num_embeddings=args.num_item, embedding_dim=1)
        self.gamma = nn.Parameter(torch.randn(args.num_aspects))  # (num_asp)

        self.loss_func = nn.MSELoss()  # loss function

        self.init_model()
    

    def init_model(self):
        def init_module(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        # emb_aspect are parameters
        nn.init.xavier_normal_(self.u_emb_aspect)
        nn.init.xavier_normal_(self.i_emb_aspect)

        # cnn
        init_module(self.im_u_cnn)
        init_module(self.im_i_cnn)

        self.ubert_out_linear.apply(init_module)
        self.ibert_out_linear.apply(init_module)

        self.ex_urev_attn_lin_tanh.apply(init_module)
        self.ex_irev_attn_lin_tanh.apply(init_module)

        self.im_urev_attn_lin_tanh.apply(init_module)
        self.im_irev_attn_lin_tanh.apply(init_module)

        self.im_u_cls_linear.apply(init_module)
        self.im_i_cls_linear.apply(init_module)

        self.ex_mlp.apply(init_module)
        self.im_mlp.apply(init_module)

        nn.init.uniform_(self.b_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.b_t.weight, -0.1, 0.1)
        nn.init.uniform_(self.gamma, -0.1, 0.1)
        

    def forward(self, batch):
        """forward function of APRE.
        What's in a `batch`?
            - u_split, i_split: split segments for each user
            - u_out_hid, i_out_hid: output hidden states for last layer. 
            - urevs_loc, irevs_loc: 
            - u_pooler, i_pooler: the CLS representation
        out_hid (last output seq. dim): tensor (ttl_nrev, pad_len, bert_dim)
                out_hid --> _rev_output_hid 
        pooler (CLS token output. dim): tensor (ttl_nrev, bert_dim)
                pooler  --> _rev_cls_hid
        """

        # ==============================
        #   getting BERT encoding
        # ==============================

        u_split, i_split = batch['u_split'], batch['i_split'] # List: [nrev, ...]

        urev_output_hid, irev_output_hid = batch['u_out_hid'], batch['i_out_hid']
        urevs_cls_hid, irevs_cls_hid = batch['u_pooler'], batch['i_pooler']
        urevs_loc, irevs_loc = batch['urevs_loc'], batch['irevs_loc']

        # [linear]
        #   Input: (ttl_nrev, pad_len, bert_dim)
        #   Output: (ttl_nrev, pad_len, feat_dim)
        urev_output_hid = self.ubert_out_linear(urev_output_hid)
        irev_output_hid = self.ibert_out_linear(irev_output_hid)

        # ===================================
        #  Process info in Ex and Im Channels
        # ===================================
        
        # Handle Explicit information
        if self.use_exp:
            # Shape of urevs_loc/irevs_loc
            #       Original: bs * [nrev [sp_mat(num_asp*pad_len)]]; 
            #       Now: (ttl_nrev) * num_asp * pad_len

            # User/Item Aspect Representation
            # [matmul]:
            #   Left:   (ttl_nrev, num_asp, pad_len) 
            #   Right:  (ttl_nrev, pad_len, feat_dim)
            #   Result: (ttl_nrev, num_asp, feat_dim)
            uasp_repr = torch.matmul(urevs_loc.view(-1, self.num_asp, self.pad_len), urev_output_hid)
            iasp_repr = torch.matmul(irevs_loc.view(-1, self.num_asp, self.pad_len), irev_output_hid)

            # [split] 
            #   Before: (see above) 
            #   After (List):  bs * tensor of (nrev, num_asp, feat_dim)
            uasp_repr_spl = torch.split(uasp_repr, u_split, dim=0)
            iasp_repr_spl = torch.split(iasp_repr, i_split, dim=0)
            
            # User/Item Review-wise Aggregation for Aspects.
            uasp_repr_agg, iasp_repr_agg = [], []
            for i in range(len(u_split)):
                # NOTE: method2 without concatenation
                num_u_rev, num_i_rev = u_split[i], i_split[i]

                # Duplicate Dim[0] of self.emb_aspect by `num_u_rev` times
                # self.emb_aspect (num_asp, feat_dim)
                #   [unsqueeze] Result: (1, num_asp, feat_dim)
                #   [expand]    Result: (num_u_rev, num_asp, feat_dim)
                ex_u_emb_aspect = torch.unsqueeze(self.u_emb_aspect, 0).expand(num_u_rev, -1, -1)

                # After cat => (nrev, num_asp, 2*feat_dim); After linear => nrev*num_asp*1
                # [OUTPUT] urev_attn_w & irev_attn_w can be output!

                # Compute Attention Weight for user-side reviews
                # [=0]: clean out aspect embeddings where no
                # [cat] on Dim-2
                #   Left:   (nrev, num_asp, feat_dim)
                #   Right:  (num_u_rev, num_asp, feat_dim)  (num_u_rev == nrev)
                #   Result: (nrev, num_asp, 2*feat_dim)
                # [sum] on Dim-2
                #   Input: (nrev, num_asp, feat_dim)
                #   Output: (nrev, num_asp, 1)  (keepdim)
                # [*]
                #   Left: (nrev, num_asp, 2*feat_dim)
                #   Right: (nrev, num_asp, 1)
                #   Result: (nrev, num_asp, 2*feat_dim)
                # [linear-tanh]+[softmax] (softmax doesn't change dimensions)
                #   Input:  (see above)
                #   Output: (nrev, num_asp, 1)
                # TODO: [tuning] whether to add this cleaning aspect line
                urev_attn_w = F.softmax(dim=0, input=self.ex_urev_attn_lin_tanh(
                    torch.cat((uasp_repr_spl[i], ex_u_emb_aspect), dim=2) * torch.sum(uasp_repr_spl[i], dim=2, keepdim=True)))

                # Apply Weight to review representations
                # [mul]
                #   Left:   (nrev, num_asp, 1)
                #   Right:  (nrev, num_asp, feat_dim)
                #   Result: (nrev, num_asp, feat_dim)
                # [sum] on Dim-0
                #   Input:  (see above)
                #   Output: (num_asp, feat_dim)
                uasp_repr_revagg = torch.sum(torch.mul(urev_attn_w, uasp_repr_spl[i]), dim=0, keepdim=False)
                
                # Save as above
                ex_i_emb_aspect = torch.unsqueeze(self.i_emb_aspect, 0).expand(num_i_rev, -1, -1)
                irev_attn_w = F.softmax(dim=0, input=self.ex_irev_attn_lin_tanh(
                    torch.cat((iasp_repr_spl[i], ex_i_emb_aspect), dim=2) * torch.sum(iasp_repr_spl[i], dim=2, keepdim=True)))
                iasp_repr_revagg = torch.sum(torch.mul(irev_attn_w, iasp_repr_spl[i]), dim=0, keepdim=False)
                
                uasp_repr_agg.append(uasp_repr_revagg)
                iasp_repr_agg.append(iasp_repr_revagg)
            
            # bs*num_asp* feat_dim
            # [stack]
            #   Before: [list] bs * [(num_asp, feat_dim)]
            #   After: (bs, num_asp, feat_dim)
            u_expl_repr = torch.stack(uasp_repr_agg, dim=0)
            i_expl_repr = torch.stack(iasp_repr_agg, dim=0)

        
        # Handle Implicit information 
        if self.use_imp:
            # Original out_hid dim: (ttl_nrev, pad_len, bert_dim)

            # Compute CNN-based Features!
            # [unsqueeze]
            #   Before: (ttl_nrev, pad_len, feat_dim)
            #   After:  (ttl_nrev, 1, pad_len, feat_dim)
            # [cnn]
            #   Input:  (see above)
            #   Output: (ttl_nrev, out_channel, cnn_len, 1)
            # [squeeze]
            #   Before: (see above)
            #   After:   (ttl_nrev, out_channel, cnn_len)
            urev_cnn_feat = F.relu(self.im_u_cnn(urev_output_hid.unsqueeze(1))).squeeze(3)
            
            # [max_pool1d]
            #   Before: (see above)
            #   After:  (ttl_nrev, out_channel, 1)
            # [squeeze]
            #   Before: (see above)
            #   After:  (ttl_nrev, out_channel)
            urev_cnn_feat = F.max_pool1d(urev_cnn_feat, urev_cnn_feat.size(2)).squeeze(2)

            # Same CNN for Item
            irev_cnn_feat = F.relu(self.im_i_cnn(irev_output_hid.unsqueeze(1))).squeeze(3)
            irev_cnn_feat = F.max_pool1d(irev_cnn_feat, irev_cnn_feat.size(2)).squeeze(2)
            
            # Compute Maxpooling and AVGpooling
            # Compute AVG pooling of sentence representation
            # [sum] & [sum] & [div]
            #   Before: (ttl_nrev, pad_len, feat_dim)
            #   After:  (ttl_nrev, feat_dim)  (div doesn't change shape)
            urevs_avgpool_repr = torch.div(torch.sum(urev_output_hid, dim=1), torch.sum(urev_output_hid!=0, dim=1))
            irevs_avgpool_repr = torch.div(torch.sum(irev_output_hid, dim=1), torch.sum(irev_output_hid!=0, dim=1))

            # Compute MAX pooling of sentence representation
            # [permute]
            #   Before: (ttl_nrev, pad_len, feat_dim)
            #   After: (ttl_nrev, feat_dim, pad_len)
            # [max_pool1d]
            #   Before: (see above)
            #   After:  (ttl_nrev, feat_dim, 1)
            #  [squeeze]
            #   Before:  (see above)
            #   After:   (ttl_nrev, feat_dim)
            urevs_maxpool_repr = torch.max_pool1d(urev_output_hid.permute(0, 2, 1), 
                urev_output_hid.size(1)).squeeze(2)
            irevs_maxpool_repr = torch.max_pool1d(irev_output_hid.permute(0, 2, 1), 
                irev_output_hid.size(1)).squeeze(2)

            # Compute CLS features
            # [linear]
            #   Input: (ttl_nrev, bert_dim)
            #   Output: (ttl_nrev, feat_dim)
            urevs_cls_hid = self.im_u_cls_linear(urevs_cls_hid)
            irevs_cls_hid = self.im_i_cls_linear(irevs_cls_hid)
            
            # Combine three different features
            # [concat]
            #   Before: [(ttl_nrev, feat_dim), (ttl_nrev, feat_dim), (ttl_nrev, feat_dim), (ttl_out_channel)
            #   After: (ttl_nrev, 3*feat_dim+out_channel)
            #   We let im_dim = 3*feat_dim+out_channel
            uconcat_repr = torch.cat(
                [urevs_cls_hid, urevs_avgpool_repr, urevs_maxpool_repr, urev_cnn_feat], dim=-1)
            # print("urevs_cls_hid", urevs_cls_hid.shape)
            # print("urevs_avgpool_repr", urevs_avgpool_repr.shape)
            # print("urevs_maxpool_repr", urevs_maxpool_repr.shape)
            # print("urev_cnn_feat", urev_cnn_feat.shape)
            iconcat_repr = torch.cat(
                [irevs_cls_hid, irevs_avgpool_repr, irevs_maxpool_repr, irev_cnn_feat], dim=-1)

            # [split]
            #   Before: see above
            #   After: [List] bs * (nrev, im_dim)
            uimpl_repr_spl = torch.split(uconcat_repr, u_split)
            iimpl_repr_spl = torch.split(iconcat_repr, i_split)

            uimpl_repr_agg, iimpl_repr_agg = [], []
            for i in range(len(u_split)):
                # [linear] + [tanh] + [softmax] - on Dim-0
                #   Before: (nrev, im_dim)
                #   After:  (nrev, 1)
                uimpl_rev_w = F.softmax(dim=0, input=self.im_urev_attn_lin_tanh(uimpl_repr_spl[i]))
                iimpl_rev_w = F.softmax(dim=0, input=self.im_irev_attn_lin_tanh(iimpl_repr_spl[i]))
                
                # [mul]
                #   Left: (nrev, im_dim)
                #   Right: (nrev, 1)
                #   Result: (nrev, im_dim)
                # [sum]
                #   Input: (see above)
                #   Output: (im_dim)
                uimpl_rev_attn_agg = torch.sum(torch.mul(uimpl_repr_spl[i], uimpl_rev_w), dim=0)
                iimpl_rev_attn_agg = torch.sum(torch.mul(iimpl_repr_spl[i], iimpl_rev_w), dim=0)
                
                uimpl_repr_agg.append(uimpl_rev_attn_agg)
                iimpl_repr_agg.append(iimpl_rev_attn_agg)
            
            # [stack]
            #   Before: bs * [(im_dim,)]
            #   After: (bs, im_dim)
            u_impl_repr = torch.stack(uimpl_repr_agg, dim=0)
            i_impl_repr = torch.stack(iimpl_repr_agg, dim=0)

        # ==========================
        #   Merge Ex & Im channels
        # ==========================
        # create tensor for user ID and item ID
        batch_uid, batch_iid = batch['uid'], batch['iid']

        # Adding up all scores!
        # [embedding]
        #   Result: (bs, 1)
        # [cat]
        #   Left: (bs, im_dim); Right: (bs, im_dim)
        #   Result: (bs, 2*im_dim)
        # [linear]
        #   Input: (see above); Output: (bs, 1)
        pred = self.b_u(batch_uid) + self.b_t(batch_iid)

        if self.use_imp:
            pred += self.im_mlp(torch.cat((u_impl_repr, i_impl_repr), dim=-1))
        
        pred = pred.squeeze()  # (bs)

        # [cat]
        #   Left: (bs, num_asp, feat_dim); Right: (bs, num_asp, feat_dim)
        #   Result: (bs, num_asp, 2*feat_dim)
        # [mlp]
        #   Input: (bs, num_asp, 2*feat_dim); Output: (bs, num_asp, 1)
        # [squeeze]
        #   Input: (see above); Output: (bs, num_asp)
        # [unsqueeze]
        #   Input: (num_asp,); Output: (num_asp, 1)
        # [matmul]
        #   Left: (bs, num_asp); Right: (num_asp, 1)
        #   Result: (bs, 1)
        # [squeeze]
        #   Input: (see above); Output: (bs)

        if self.use_exp:
            pred += torch.matmul(
                self.ex_mlp(torch.cat((u_expl_repr, i_expl_repr), dim=-1)).squeeze(), 
                torch.unsqueeze(self.gamma, 1)).squeeze()

        return pred