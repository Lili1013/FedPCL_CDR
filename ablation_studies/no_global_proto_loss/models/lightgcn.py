from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Light_GCN(nn.Module):
    def __init__(self,**params):
        super(Light_GCN, self).__init__()
        self.device = params['device']
        # self.u_emb = params['u_emb']
        # self.v_emb = params['v_emb']
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.n_nodes = self.user_num + self.item_num
        self.train_data = params['train_data']
        self.n_layers = params['n_layers']
        self.mat = self.create_sparse_matrix(self.train_data, self.user_num, self.item_num)
        self.norm_adj = self.get_norm_adj_mat(self.mat.astype(np.float32), self.user_num,
                                                     self.item_num, self.n_nodes).to(self.device)
        # self.u_g_embeddings, self.v_g_embeddings = self.get_user_item_id_emb(self.u_emb,self.v_emb,self.user_num,self.item_num,self.norm_adj)

    def create_sparse_matrix(self, df_feat, user_num,item_num,form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat['userID'].values
        tgt = df_feat['itemID'].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def get_norm_adj_mat(self, interaction_matrix,user_num,item_num,n_nodes):
        A = sp.dok_matrix((user_num + item_num,
                           user_num + item_num), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + user_num),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + user_num, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((n_nodes, n_nodes)))

    def get_user_item_id_emb(self,u_emb,v_emb):

        h = v_emb.weight

        ego_embeddings = torch.cat((u_emb.weight, v_emb.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        all_embeddings = all_embeddings.sum(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], dim=0)
        return u_g_embeddings, i_g_embeddings