
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightgcn import Light_GCN




class Local_Model(nn.Module):
    def __init__(self, **params):
        super(Local_Model, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.tau = params['tau']
        self.embed_id_dim = params['embed_id_dim']
        # self.disen_emb_dim = params['disen_emb_dim']
        self.device = params['device']
        self.n_layers = params['n_layers']
        self.train_data = params['train_data']
        self.review_embed_dim = params['review_embed_dim']
        self.u_review_feat = params['u_review_feat']
        self.v_review_feat = params['v_review_feat']
        # self.disen_feat_agg_way = params['disen_feat_agg_way']
        self.u_emb = nn.Embedding(self.user_num, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.u_emb.weight)
        self.v_emb = nn.Embedding(self.item_num, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.v_emb.weight)
        self.u_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.u_review_feat_emb.weight.data.copy_(self.u_review_feat)
        self.u_review_feat_emb.weight.requires_grad = False
        self.v_review_feat_emb = nn.Embedding(self.item_num, self.review_embed_dim).to(self.device)
        self.v_review_feat_emb.weight.data.copy_(self.v_review_feat)
        self.v_review_feat_emb.weight.requires_grad = False
        self.u_review_feat_layer = nn.Linear(self.review_embed_dim,self.embed_id_dim).to(self.device)
        self.v_review_feat_layer = nn.Linear(self.review_embed_dim,self.embed_id_dim).to(self.device)

        light_gcn_params = {
            'device': self.device,
            # 'u_emb': self.u_emb,
            # 'v_emb': self.v_emb,
            'user_num': self.user_num,
            'item_num': self.item_num,
            'train_data': self.train_data,
            'n_layers': self.n_layers
        }
        self.light_gcn = Light_GCN(**light_gcn_params)
        # self.u_id_embeddings, self.v_id_embeddings = self.light_gcn.u_g_embeddings, self.light_gcn.v_g_embeddings
        # disen_params = {
        #     'orig_emb_size': self.embed_id_dim,
        #     'disen_emb_size': self.disen_emb_dim,
        #     'device': self.device
        # }
        # self.domain_disen_model = Domain_Disen(**disen_params)

        # # #the aggregation layers for id and review features
        # self.v_feat_cat_layer = nn.Linear(self.embed_id_dim*2, self.embed_id_dim).to(self.device)
        # self.v_feat_cat_norm = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        # self.u_feat_cat_layer = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)
        # self.u_feat_cat_norm = nn.BatchNorm1d(self.embed_id_dim).to(self.device)

        # if self.disen_feat_agg_way == 'concat':
        #     self.disen_agg_layer = nn.Linear(self.disen_emb_dim*2,self.disen_emb_dim).to(self.device)
        #     self.disen_agg_norm = nn.BatchNorm1d(self.disen_emb_dim).to(self.device)

        # self.v_feat_layer = nn.Linear(self.embed_id_dim,self.disen_emb_dim).to(self.device)
        # self.v_feat_norm = nn.BatchNorm1d(self.disen_emb_dim).to(self.device)

        self.inter_learn_layer1 = nn.Linear(self.embed_id_dim*2, self.embed_id_dim).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer1.weight)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        self.inter_learn_layer2 = nn.Linear(self.embed_id_dim, self.embed_id_dim // 2).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer2.weight)
        self.batch_norm2 = nn.BatchNorm1d(self.embed_id_dim//2).to(self.device)
        self.inter_learn_layer3 = nn.Linear(self.embed_id_dim//2, self.embed_id_dim // 4).to(self.device)
        nn.init.xavier_uniform_(self.inter_learn_layer3.weight)
        self.batch_norm3 = nn.BatchNorm1d(self.embed_id_dim // 4).to(self.device)
        self.classifier_layer = nn.Linear(self.embed_id_dim // 4, 1).to(self.device)

        # loss function
        self.criterion = nn.BCELoss()

    def forward(self, nodes_u, nodes_v):
        self.u_id_embeddings,self.v_id_embeddings = self.light_gcn.get_user_item_id_emb(self.u_emb,self.v_emb)
        u_id_feats = self.u_id_embeddings[nodes_u]
        v_id_feats = self.v_id_embeddings[nodes_v]

        self.u_rev_embeddings, self.v_rev_embeddings = self.light_gcn.get_user_item_id_emb(self.u_review_feat_emb, self.v_review_feat_emb)
        u_review_feats = self.u_rev_embeddings[nodes_u]
        v_review_feats = self.v_rev_embeddings[nodes_v]
        u_review_feats = F.relu(self.u_review_feat_layer(u_review_feats))
        v_review_feats = F.relu(self.v_review_feat_layer(v_review_feats))
        # u_review_feats = self.u_review_feat_emb(nodes_u)
        # v_review_feats = self.v_review_feat_emb(nodes_v)
        # u_feats = torch.cat([u_id_feats,u_review_feats],dim=1)
        # v_feats = torch.cat([v_id_feats,v_review_feats],dim=1)
        # u_feats = self.u_feat_cat_norm(F.relu(self.u_feat_cat_layer(u_feats)))
        # v_feats = self.v_feat_cat_norm(F.relu(self.v_feat_cat_layer(v_feats)))
        u_feats = u_review_feats+u_id_feats
        v_feats = v_review_feats+v_id_feats
        # u_common_feats, u_specific_feats = self.domain_disen_model.forward(u_feats)
        # v_feats = self.v_feat_norm(F.relu(self.v_feat_layer(v_feats)))
        # agg way: sum
        # if self.disen_feat_agg_way == 'sum':
        #     u_feats = u_common_feats + u_specific_feats
        # elif self.disen_feat_agg_way == 'avg':
        #     u_feats = (u_common_feats+u_specific_feats)/2
        # elif self.disen_feat_agg_way == 'concat':
        #     u_feats = self.disen_agg_norm(F.relu(self.disen_agg_layer(torch.cat([u_common_feats,u_specific_feats],dim=1))))

        inter_feats = torch.cat([u_feats, v_feats], dim=1)
        inter_feats1 = self.batch_norm1(F.relu(self.inter_learn_layer1(inter_feats)))
        inter_feats2 = self.batch_norm2(F.relu(self.inter_learn_layer2(inter_feats1)))
        inter_feats3 = self.batch_norm3(F.relu(self.inter_learn_layer3(inter_feats2)))
        pred_prob = F.sigmoid(self.classifier_layer(inter_feats3))
        return u_feats, v_feats, pred_prob.squeeze()

    def cross_entropy_loss(self, pred_labels, true_labels):
        L_ce = self.criterion(pred_labels, true_labels.to(torch.float32))
        return L_ce

    def info_nce_loss(self, anchor, positive, negatives, temperature=1.0):
        # Calculate cosine similarity between anchor and positive
        cos_sim = F.cosine_similarity(anchor, positive, dim=1)

        # Calculate cosine similarity between anchor and negatives
        cos_sim_neg = F.cosine_similarity(anchor.unsqueeze(1).expand_as(negatives), negatives, dim=2)

        # Concatenate positive and negative similarities
        logits = torch.cat([cos_sim.unsqueeze(1), cos_sim_neg], dim=1)

        # Apply temperature to the logits
        logits /= temperature

        # Calculate softmax probabilities
        probs = F.softmax(logits, dim=1)

        # InfoNCE loss
        info_nce_loss = -torch.log(probs[:, 0]).mean()

        return info_nce_loss, probs


    def proto_loss(self, P, C, nodes_u, category_lists, u_feats, cat_labels,client_num):
        if client_num == 3:
            L_C, L_P = self.proto_loss_multiple(P, C, nodes_u, category_lists, u_feats, cat_labels)
        else:
            L_C, L_P = self.proto_loss_dual(P, C, nodes_u, category_lists, u_feats, cat_labels)
        return L_C, 0

    def proto_loss_multiple(self,P, C, nodes_u, category_lists, u_feats, cat_labels):
        batch_size = len(nodes_u)
        c_pos_features = torch.zeros_like(u_feats).to(self.device)  # (256,32)
        c_neg_features_lists = []
        p_pos_features_1 = torch.zeros_like(u_feats).to(self.device)
        p_pos_features_2 = torch.zeros_like(u_feats).to(self.device)
        p_pos_features_3 = torch.zeros_like(u_feats).to(self.device)
        p_neg_features_lists = []
        cat_num = len(cat_labels)
        cat_lists = list(cat_labels)
        for i in range((cat_num - 1) * 3):
            if i < cat_num - 1:
                c_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))
                p_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))
            else:
                p_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))

        for i in range(batch_size):
            c_pos_features[i, :] = torch.tensor(C[int(category_lists[i])]).to(self.device)
            p_pos_features_1[i, :] = torch.tensor(P[int(category_lists[i])][0]).to(self.device)
            try:
                p_pos_features_2[i, :] = torch.tensor(P[int(category_lists[i])][1]).to(self.device)
                p_pos_features_3[i, :] = torch.tensor(P[int(category_lists[i])][2]).to(self.device)
            except:
                p_pos_features_2[i, :] = torch.tensor(P[int(category_lists[i])][0]).to(self.device)
                p_pos_features_3[i, :] = torch.tensor(P[int(category_lists[i])][0]).to(self.device)
            cat_lists_temp = cat_lists
            cat_filter_lists = list(set(cat_lists_temp) - set([int(category_lists[i])]))
            for j in range(len(c_neg_features_lists)):
                neg_cat = cat_filter_lists[0]
                c_neg_features_lists[j][i, :] = torch.tensor(C[neg_cat]).to(self.device)
                p_neg_features_lists[j][i, :] = torch.tensor(P[neg_cat][0]).to(self.device)
                try:
                    p_neg_features_lists[j + cat_num - 1][i, :] = torch.tensor(P[neg_cat][1]).to(self.device)
                    p_neg_features_lists[j + (cat_num - 1) * 2][i, :] = torch.tensor(P[neg_cat][2]).to(self.device)
                except:
                    p_neg_features_lists[j + cat_num - 1][i, :] = torch.tensor(P[neg_cat][0]).to(self.device)
                    p_neg_features_lists[j + (cat_num - 1) * 2][i, :] = torch.tensor(P[neg_cat][0]).to(self.device)

                cat_filter_lists = [item for item in cat_filter_lists if item != neg_cat]
        negatives_p = torch.stack(p_neg_features_lists, dim=1)
        negatives_c = torch.stack(c_neg_features_lists, dim=1)
        # calculate L_P
        L_C, prob_c = self.info_nce_loss(anchor=u_feats, positive=c_pos_features, negatives=negatives_c,
                                         temperature=self.tau)
        _, prob_p_1 = self.info_nce_loss(anchor=u_feats, positive=p_pos_features_1, negatives=negatives_p,
                                         temperature=self.tau)
        _, prob_p_2 = self.info_nce_loss(anchor=u_feats, positive=p_pos_features_2, negatives=negatives_p,
                                         temperature=self.tau)
        _, prob_p_3 = self.info_nce_loss(anchor=u_feats, positive=p_pos_features_2, negatives=negatives_p,
                                         temperature=self.tau)
        L_P = -torch.log(prob_p_1[:, 0] + prob_p_2[:, 0] + prob_p_3[:, 0]).mean()
        return L_C, L_P

    def proto_loss_dual(self,P, C, nodes_u, category_lists, u_feats, cat_labels):
        batch_size = len(nodes_u)
        c_pos_features = torch.zeros_like(u_feats).to(self.device)  # (256,32)
        c_neg_features_lists = []
        p_pos_features_1 = torch.zeros_like(u_feats).to(self.device)
        p_pos_features_2 = torch.zeros_like(u_feats).to(self.device)
        p_neg_features_lists = []
        cat_num = len(cat_labels)
        cat_lists = list(cat_labels)
        for i in range((cat_num - 1) * 2):
            if i < cat_num - 1:
                c_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))
                p_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))
            else:
                p_neg_features_lists.append(torch.zeros_like(u_feats).to(self.device))

        for i in range(batch_size):
            c_pos_features[i, :] = torch.tensor(C[int(category_lists[i])]).to(self.device)
            p_pos_features_1[i, :] = torch.tensor(P[int(category_lists[i])][0]).to(self.device)
            try:
                p_pos_features_2[i, :] = torch.tensor(P[int(category_lists[i])][1]).to(self.device)

            except:
                p_pos_features_2[i, :] = torch.tensor(P[int(category_lists[i])][0]).to(self.device)

            cat_lists_temp = cat_lists
            cat_filter_lists = list(set(cat_lists_temp) - set([int(category_lists[i])]))
            for j in range(len(c_neg_features_lists)):
                neg_cat = cat_filter_lists[0]
                c_neg_features_lists[j][i, :] = torch.tensor(C[neg_cat]).to(self.device)
                p_neg_features_lists[j][i, :] = torch.tensor(P[neg_cat][0]).to(self.device)
                try:
                    p_neg_features_lists[j + cat_num - 1][i, :] = torch.tensor(P[neg_cat][1]).to(self.device)
                except:
                    p_neg_features_lists[j + cat_num - 1][i, :] = torch.tensor(P[neg_cat][0]).to(self.device)
                cat_filter_lists = [item for item in cat_filter_lists if item != neg_cat]
        negatives_p = torch.stack(p_neg_features_lists, dim=1)
        negatives_c = torch.stack(c_neg_features_lists, dim=1)
        # calculate L_P
        L_C, prob_c = self.info_nce_loss(anchor=u_feats, positive=c_pos_features, negatives=negatives_c,
                                         temperature=self.tau)
        _, prob_p_1 = self.info_nce_loss(anchor=u_feats, positive=p_pos_features_1, negatives=negatives_p,
                                         temperature=self.tau)
        _, prob_p_2 = self.info_nce_loss(anchor=u_feats, positive=p_pos_features_2, negatives=negatives_p,
                                         temperature=self.tau)

        L_P = -torch.log(prob_p_1[:, 0] + prob_p_2[:, 0]).mean()
        return L_C, L_P


    def disc_loss(self, x, y, kernel="rbf"):
        """Emprical maximum mean discrepancy. The lower the result
         the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device))
        '''
        XX, YY, XY = (torch.zeros(xx.shape),
                      torch.zeros(xx.shape),
                      torch.zeros(xx.shape))
        '''

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1
        elif kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)
        return -torch.mean(XX + YY - 2. * XY)






