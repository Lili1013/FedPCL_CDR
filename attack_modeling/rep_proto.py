import os
import sys
curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.load_data import Load_Data
from utils.para_parser import parse
from models.local_model import Local_Model
from utils.params import data_params,model_params
from models.kmeans_ldp import k_means,spectral_cluster,gmm_clustering

from utils.evaluation import calculate_hr_ndcg
from utils.utils import calculate_cos_sim

import os
import torch
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def add_laplace_noise(data,noise_control):
    # self.beta = self.sensitivity/self.epsilon
    noise = np.random.laplace(0, noise_control, data.shape)
    # noise = torch.tensor(np.random.normal(0, noise_control, data.shape))
    # noise = noise.to(torch.float32)
    # noisy_data = data+noise
    return noise


def local_model_update(args, P, C, model, local_data,optimizer,device,round_number,client_number,cat_labels):
    # logger.info('start train')
    model.eval()

    # total_client_loss, total_client_ce_loss, total_client_p_loss, total_client_c_loss, total_client_disc_loss = [], [], [], [], []
    # for iter in range(args.epochs):
    #     # logger.info('epoch : {}',iter)
    #     total_loss,total_ce_loss,total_p_loss,total_c_loss, total_disc_loss= [],[],[],[],[]
    #     for i, data in enumerate(local_data.train_loader, 0):
    #         if i >= 1:
    #             break
    #         batch_nodes_u,batch_nodes_v,batch_nodes_c,batch_nodes_r = data
    #         batch_nodes_u, batch_nodes_v, categories_list, labels_list = local_data.generate_train_instances(batch_nodes_u.tolist(),
    #                                                                                         batch_nodes_v.tolist(),batch_nodes_c.tolist())
    #         batch_nodes_u, batch_nodes_v, categories_list,labels_list = torch.tensor(batch_nodes_u), torch.tensor(batch_nodes_v), torch.tensor(categories_list,dtype=int),torch.tensor(labels_list)
    #         optimizer.zero_grad()
    #         u_feats, v_feats, pred_prob = model.forward(batch_nodes_u.to(device), batch_nodes_v.to(device))
    #         L_ce = model.cross_entropy_loss(pred_prob,labels_list.to(device))
    #         # L_disc = model.disc_loss(u_common_feats,u_specific_feats)
    #         if len(P)==0 or len(C)==0:
    #             L_P = 0*L_ce
    #             L_C = 0*L_ce
    #         else:
    #             L_C,L_P = model.proto_loss(P,C,batch_nodes_u,categories_list.to(device),u_feats,cat_labels,args.client_num)
    #             # L_P, L_C = model.calculate_proto_loss(P, C, batch_nodes_u.tolist(), categories_list.tolist(), u_feats.to(torch.device('cpu')).detach().numpy())
    #         L = L_ce+args.alpha*(L_C+L_P)
    #         # logger.info('loss cross entropy: {}, loss P: {}, loss C:{}',L_ce.item(),L_P.item(),L_C.item())
    #         total_loss.append(L.item())
    #         total_ce_loss.append(L_ce.item())
    #         total_c_loss.append(L_C.item())
    #         total_p_loss.append(L_P.item())
    #         total_disc_loss.append(0)
    # 
    #         L.backward(retain_graph=True)
    #         optimizer.step()
    #         # logger.info('loss:{}',L.item())
    #     total_client_loss.append(sum(total_loss) / len(total_loss))
    #     total_client_disc_loss.append(sum(total_disc_loss)/len(total_disc_loss))
    #     total_client_ce_loss.append(sum(total_ce_loss) / len(total_ce_loss))
    #     total_client_p_loss.append(sum(total_p_loss) / len(total_p_loss))
    #     total_client_c_loss.append(sum(total_c_loss) / len(total_c_loss))

    nodes_u = torch.tensor(list(local_data.train_df['userID'].unique())).to(device)
    u_id_embeddings, v_id_embeddings = model.light_gcn.get_user_item_id_emb(model.u_emb, model.v_emb)
    u_rev_embeddings, v_rev_embeddings = model.light_gcn.get_user_item_id_emb(model.u_review_feat_emb,
                                                                              model.v_review_feat_emb)
    # nodes_v = torch.tensor(list(local_data.train_df['itemID'].unique())).to(device)

    model.eval()
    with torch.no_grad():
        u_id_feats = u_id_embeddings[nodes_u]
        u_rev_feats = u_rev_embeddings[nodes_u]
        u_rev_feats = F.relu(model.u_review_feat_layer(u_rev_feats))
        u_feats = u_id_feats + u_rev_feats
        # u_common_feats, u_specific_feats = model.domain_disen_model.forward(u_feats)
    u_feats = u_feats.cpu().detach().numpy()
    overlap_users = local_data.overlap_users

    cat_dict, cluster_overlap_smaples, labels = k_means(num_clusters=args.num_clusters, embeddings=u_feats,
                                                        overlap_users=overlap_users,args=args)
    # cat_dict, cluster_overlap_smaples, labels = spectral_cluster(num_clusters=args.num_clusters, embeddings=u_feats,
    #                                                     overlap_users=overlap_users)
    # cat_dict, cluster_overlap_smaples, labels = gmm_clustering(num_clusters=args.num_clusters, embeddings=u_feats,
    #                                                              overlap_users=overlap_users)
    local_data.update_cat(category_dict=cat_dict)
    return cluster_overlap_smaples

def server_update(P,C,local_data_list,overlap_user_num,global_proto_agg_way):
    client_num = len(local_data_list)
    P_new = {}
    if client_num == 3:
        df_0 = local_data_list[0].train_df[(local_data_list[0].train_df['userID']>=0)
                                           &(local_data_list[0].train_df['userID']<overlap_user_num)].groupby('userID').first().reset_index()[['userID','category']]
        df_1 = local_data_list[1].train_df[(local_data_list[1].train_df['userID'] >= 0)
                                           & (local_data_list[1].train_df['userID'] < overlap_user_num)].groupby(
            'userID').first().reset_index()[['userID', 'category']]
        df_2 = local_data_list[2].train_df[(local_data_list[2].train_df['userID'] >= 0)
                                           & (local_data_list[2].train_df['userID'] < overlap_user_num)].groupby(
            'userID').first().reset_index()[['userID', 'category']]
        df = pd.concat([df_0[['userID','category']].rename(columns={'userID':'userID','category':'category_0'}),
                        df_1[['category']].rename(columns={'category':'category_1'}),
                        df_2[['category']].rename(columns={'category':'category_2'})
                        ],axis=1)
    else:
        df_0 = local_data_list[0].train_df[(local_data_list[0].train_df['userID'] >= 0)
                                           & (local_data_list[0].train_df['userID'] < overlap_user_num)].groupby(
            'userID').first().reset_index()[['userID', 'category']]
        df_1 = local_data_list[1].train_df[(local_data_list[1].train_df['userID'] >= 0)
                                           & (local_data_list[1].train_df['userID'] < overlap_user_num)].groupby(
            'userID').first().reset_index()[['userID', 'category']]
        df = pd.concat([df_0[['userID', 'category']].rename(
            columns={'userID': 'userID', 'category': 'category_0'}),
                        df_1[['category']].rename(
                            columns={'category': 'category_1'})
                        ], axis=1)
    if global_proto_agg_way == 'avg' or global_proto_agg_way == 'sum':
        P_new,C = update_protos_avg_sum(P,C,df,client_num,global_proto_agg_way)
    elif global_proto_agg_way == 'weight':
        P_new,C = update_protos_weight(P,C,df,client_num)
    return P_new,C

def update_protos_avg_sum(P,C,df,client_num,global_proto_agg_way):
    # noise = np.random.laplace(0, 0.2, np.array(P[0][0][0]).shape)
    P_new = {}
    P_all = {}
    for client, client_value in P.items():
        P_new[client] = {}
        P_all[client] = {}
        for cat, cat_value in client_value.items():
            P_new[client][cat] = []
            P_all[client][cat] = []
            indexes = cat_value[2]
            anchor = cat_value[1]
            # anchor = list(anchor+noise)
            if len(indexes) == 0:
                P_new[client][cat].append(anchor)
                P_all[client][cat].append(anchor)
                continue
            df_result = df.loc[indexes]
            if client_num == 3:
                client_0_cats = set(df_result['category_0'].values)
                client_1_cats = set(df_result['category_1'].values)
                client_2_cats = set(df_result['category_2'].values)
                sim = float('-inf')
                for each_cat in client_0_cats:
                    P_all[client][cat].append(P[0][each_cat][1])
                    sim_0 = calculate_cos_sim(anchor, P[0][each_cat][1])
                    if sim_0 > sim:
                        sim = sim_0
                        cat_0 = each_cat

                P_new[client][cat].append(P[0][cat_0][1])
                sim = float('-inf')
                for each_cat in client_1_cats:
                    P_all[client][cat].append(P[1][each_cat][1])
                    sim_1 = calculate_cos_sim(anchor, P[1][each_cat][1])
                    if sim_1 > sim:
                        sim = sim_1
                        cat_1 = each_cat

                P_new[client][cat].append(P[1][cat_1][1])
                sim = float('-inf')
                for each_cat in client_2_cats:
                    P_all[client][cat].append(P[2][each_cat][1])
                    sim_2 = calculate_cos_sim(anchor, P[2][each_cat][1])
                    if sim_2 > sim:
                        sim = sim_2
                        cat_2 = each_cat
                P_new[client][cat].append(P[2][cat_2][1])

            else:
                client_0_cats = set(df_result['category_0'].values)
                client_1_cats = set(df_result['category_1'].values)
                sim = float('-inf')
                for each_cat in client_0_cats:
                    P_all[client][cat].append(list(P[0][each_cat][1]))
                    sim_0 = calculate_cos_sim(anchor, list(P[0][each_cat][1]))
                    if sim_0 > sim:
                        sim = sim_0
                        cat_0 = each_cat
                # logger.info(f'client:{client}, cat:{cat},cat_0:{cat_0},sim:{sim}')
                P_new[client][cat].append(list(P[0][cat_0][1]))
                sim = float('-inf')
                for each_cat in client_1_cats:
                    P_all[client][cat].append(list(P[1][each_cat][1]))
                    sim_1 = calculate_cos_sim(anchor, list(P[1][each_cat][1]))
                    if sim_1 > sim:
                        sim = sim_1
                        cat_1 = each_cat
                # logger.info(f'client:{client}, cat:{cat},cat_0:{cat_1},sim:{sim}')
                P_new[client][cat].append(list(P[1][cat_1][1]))

    for client, client_value in P_all.items():
        C[client] = {}
        for cat, cat_value in client_value.items():
            cat_value = np.array([list(each_value) for each_value in cat_value])
            if len(cat_value) > 0:
                # avg
                if global_proto_agg_way == 'avg':
                    mean_value = np.mean(cat_value, axis=0)
                    C[client][cat] = list(mean_value)
                elif global_proto_agg_way == 'sum':
                    # sum
                    sum_value = np.sum(cat_value, axis=0)
                    C[client][cat] = list(sum_value)
            else:
                C[client][cat] = []
    return P_new, C
def update_protos_weight(P,C,df,client_num):
    P_new = {}
    P_all = {}
    for client, client_value in P.items():
        P_new[client] = {}
        P_all[client] = {}
        for cat, cat_value in client_value.items():
            P_new[client][cat] = []
            P_all[client][cat] = []
            indexes = cat_value[1]
            anchor = cat_value[0]
            cat_sample_num = cat_value[2]
            if len(indexes) == 0:
                P_new[client][cat].append(anchor)
                temp = anchor + cat_sample_num
                P_all[client][cat].append(temp)
                continue
            df_result = df.loc[indexes]
            if client_num == 3:
                client_0_cats = set(df_result['category_0'].values)
                client_1_cats = set(df_result['category_1'].values)
                client_2_cats = set(df_result['category_2'].values)
                sim = float('-inf')
                for each_cat in client_0_cats:
                    cat_sample_num = P[0][each_cat][2]
                    temp = P[0][each_cat][0] + cat_sample_num
                    P_all[client][cat].append(temp)
                    sim_0 = calculate_cos_sim(anchor, P[0][each_cat][0])
                    if sim_0 > sim:
                        sim = sim_0
                        cat_0 = each_cat
                P_new[client][cat].append(P[0][cat_0][0])
                sim = float('-inf')
                for each_cat in client_1_cats:
                    cat_sample_num = P[1][each_cat][2]
                    temp = P[1][each_cat][0] + cat_sample_num
                    P_all[client][cat].append(temp)
                    sim_1 = calculate_cos_sim(anchor, P[1][each_cat][0])
                    if sim_1 > sim:
                        sim = sim_1
                        cat_1 = each_cat
                P_new[client][cat].append(P[1][cat_1][0])
                sim = float('-inf')
                for each_cat in client_2_cats:
                    cat_sample_num = P[2][each_cat][2]
                    temp = P[2][each_cat][0] + cat_sample_num
                    P_all[client][cat].append(temp)
                    sim_2 = calculate_cos_sim(anchor, P[2][each_cat][0])
                    if sim_2 > sim:
                        sim = sim_2
                        cat_2 = each_cat
                P_new[client][cat].append(P[2][cat_2][0])

            else:
                client_0_cats = set(df_result['category_0'].values)
                client_1_cats = set(df_result['category_1'].values)
                sim = float('-inf')
                for each_cat in client_0_cats:
                    P_all[client][cat].append(P[0][each_cat][0])
                    sim_0 = calculate_cos_sim(anchor, P[0][each_cat][0])
                    if sim_0 > sim:
                        sim = sim_0
                        cat_0 = each_cat
                P_new[client][cat].append(P[0][cat_0][0])
                sim = float('-inf')
                for each_cat in client_1_cats:
                    P_all[client][cat].append(P[1][each_cat][0])
                    sim_1 = calculate_cos_sim(anchor, P[1][each_cat][0])
                    if sim_1 > sim:
                        sim = sim_1
                        cat_1 = each_cat
                P_new[client][cat].append(P[1][cat_1][0])

    for client, client_value in P_all.items():
        C[client] = {}
        for cat, cat_value in client_value.items():
            cat_value = np.array([list(each_value) for each_value in cat_value])
            sample_nums = cat_value[:, -1]
            sample_sum = np.sum(cat_value[:, -1])
            sample_sum_expanded = np.expand_dims(sample_nums, axis=1)
            cat_value = cat_value[:, 0:-1] / sample_sum * sample_sum_expanded
            if len(cat_value) > 0:
                # avg
                # mean_value = np.mean(cat_value, axis=0)
                # C[client][cat] = list(mean_value)
                # sum
                sum_value = np.sum(cat_value, axis=0)
                C[client][cat] = list(sum_value)
            else:
                C[client][cat] = []
    return P_new, C

def get_rep_proto():
    args = parse()
    logger.info(f'parameter settings: batch_size:{args.batch_size},lr:{args.lr},embed_id_dim:{args.embed_id_dim},'
                f'review_emb_dim:{args.review_embed_dim},alpha:{args.alpha},tau:{args.tau}, C:{args.C}, noise:{args.lap_noise},client_num:{args.client_num},'
                f'client_names:{args.client_names},num_clusters:{args.num_clusters},'
                f'disen_feat_agg_way:{args.disen_feat_agg_way},global_proto_agg_way:{args.global_proto_agg_way},top_k:{args.top_k},'
                f'local epochs:{args.epochs}')
    dataset = args.client_names[0].split('_')[0]
    date = f'{datetime.now().month}_{datetime.now().day}'
    train_model_weights_path, save_model_path = [], []
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(1)
    local_model_list = []
    local_data_list = []
    local_optimizer_list = []
    # save_model_paths = args.save_model_path
    # initializa local models and local data objects

    # best_model_list = [
    #     '../best_model_ldp_1/amazon/amazon_phone_sport_phone_emb_64_rev_emb_64_c_10_alpha_0.01_tau_0.2_lr_0.001_b_512_k_10_epoch_1_noise_0.1_C_0.2_11_14.model',
    #     '../best_model_ldp_1/amazon/amazon_phone_sport_sport_emb_64_rev_emb_64_c_10_alpha_0.01_tau_0.2_lr_0.001_b_512_k_10_epoch_1_noise_0.1_C_0.2_11_14.model']
    best_model_list = [
        '../best_model_ldp_1/douban/douban_movie_music_movie_emb_64_rev_emb_64_c_10_alpha_0.01_tau_0.2_lr_0.001_b_512_k_10_epoch_1_noise_0.1_C_0.2_3_17.model',
        '../best_model_ldp_1/douban/douban_movie_music_music_emb_64_rev_emb_64_c_10_alpha_0.01_tau_0.2_lr_0.001_b_512_k_10_epoch_1_noise_0.1_C_0.2_3_17.model',
    ]
    logger.info('initialize local data and local model:{}', args.client_names)
    for j in range(args.client_num):
        data_params[args.client_names[j]].update({'args': args})
        local_data = Load_Data(**data_params[args.client_names[j]])
        local_data_list.append(local_data)
        client_name = args.client_names[j].split('_')[0] + '_' + args.client_names[j].split('_')[-1]
        model_params[client_name].update({
            'tau': args.tau,
            'embed_id_dim': args.embed_id_dim,
            'device': device,
            'train_data': local_data.train_df,
            'review_embed_dim': args.review_embed_dim,
            'u_review_feat': local_data.user_review_emb,
            'v_review_feat': local_data.item_review_emb,
            'n_layers': args.n_layers,
            # 'disen_emb_dim':args.disen_embed_dim,
            'disen_feat_agg_way': args.disen_feat_agg_way
        })
        # logger.info(client_name)
        local_model = Local_Model(**model_params[client_name])
        pretrain_dict = torch.load(f'{best_model_list[j]}', map_location=lambda storage, loc: storage)  # update net parameters
        para_dict = local_model.state_dict()
        same_para_dict = {k: v for k, v in pretrain_dict.items() if k in para_dict}
        para_dict.update(same_para_dict)
        local_model.load_state_dict(para_dict)
        local_model_list.append(local_model.to(device))
        local_optimizer_list.append(
            torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.l2_regularization))

    P, P_new = {}, {}  # local prototypes for each client
    C, C_new = {}, {}  # global prototypes
    for i in range(args.client_num):
        P[i] = {}
        P_new[i] = {}
        C_new[i] = {}
        C[i] = {}
    labels = set()


    for j in range(args.client_num):
            # logger.info('round: {}, client {}',i,j)
            # logger.info('start train')
            client_P = local_model_update(args=args,
                                                                                 P=P_new[j], C=C_new[j],
                                                                                 model=local_model_list[j],
                                                                                 local_data=local_data_list[j],
                                                                                 optimizer=local_optimizer_list[j],
                                                                                 device
                                                                                 =device, round_number=i,
                                                                                 client_number=j, cat_labels=labels)
            P[j] = client_P
    return local_model_list, P







