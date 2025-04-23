import argparse

def parse():
    parser = argparse.ArgumentParser(description='FedPCL_MDR')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_id_dim', type=int, default=64, metavar='N', help='gnn embedding size')
    parser.add_argument('--disen_embed_dim', type=int, default=64, metavar='N', help='disen feature embedding size')
    parser.add_argument('--review_embed_dim', type=int, default=64, metavar='N', help='review feature embedding size')
    parser.add_argument('--n_layers', type=int, default=3, metavar='N', help='the number of GNN layers')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2_regularization', type=float, default=0.0001, metavar='weight decay', help='the weight decay of optimizer')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train')
    parser.add_argument('--rounds', type=int, default=500, metavar='N', help='number of rounds for the communication between the server and clients')
    parser.add_argument('--alpha', type=float, default=0.01, metavar='N',
                        help='hyper parameter of contrastive loss')
    parser.add_argument('--beta', type=float, default=0.01, metavar='N',
                        help='hyper parameter of mmd loss')
    parser.add_argument('--train_neg_num', type=int, default=1, metavar='N', help='the number of training negative sample')
    parser.add_argument('--test_neg_num', type=int, default=99, metavar='N', help='the number of testing negative sample')
    parser.add_argument('--top_k', type=int, default=10, metavar='N', help='the length of recommendation lists')
    parser.add_argument('--client_num', type=int, default=3, metavar='N', help='the number of clients/domains')
    parser.add_argument('--lap_noise', type=float, default=0.1, metavar='N',
                        help='the laplace noise, eta')
    parser.add_argument('--C', type=float, default=0.2, metavar='N',
                        help='the clipping threshold of sensitivity')
    # parser.add_argument('--client_names',type=list,default = ['douban_book','douban_movie','douban_music'],metavar='N', help='the clients')
    parser.add_argument('--client_names', type=list, default=['amazon_elec_cloth_phone_elec', 'amazon_elec_cloth_phone_cloth','amazon_elec_cloth_phone_phone'],
                        metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list,
    #                     default=['amazon_phone_sport_phone', 'amazon_phone_sport_sport'],
    #                     metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list,
    #                     default=['douban_book_movie_book', 'douban_book_movie_movie'],
    #                     metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list,
    #                     default=['douban_movie_music_movie', 'douban_movie_music_music'],
    #                     metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list,
    #                     default=['amazon_phone_elec_phone', 'amazon_phone_elec_elec'],
    #                     metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list,
    #                     default=['amazon_cloth_sport_cloth', 'amazon_cloth_sport_sport'],
    #                     metavar='N', help='the clients')
    parser.add_argument('--tau', type=float, default=0.2,
                        metavar='N', help='the temperature of contrastive loss')
    parser.add_argument('--num_clusters', type=int, default=10,
                        metavar='N', help='the centroids of kmeans')
    parser.add_argument('--cluster_method', type=str, default='kmeans',
                        metavar='N', help='the cluster method')
    parser.add_argument('--disen_feat_agg_way', type=str, default='sum',
                        metavar='N', help='the aggregation way for disentangled features')
    parser.add_argument('--global_proto_agg_way', type=str, default='avg',
                        metavar='N', help='the aggregation way for global prototypes')

    # parser.add_argument('--save_model_path', type=list,
    #             default=['best_models_02_23/douban/book_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #                      'best_models_02_23/douban/movie_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #                      'best_models_02_23/douban/music_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model'])
    # parser.add_argument('--train_model_weights_path', type=list,
    # default=['best_models_02_23/douban/train_model_weights_epochs_douban_book_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #         'best_models_02_23/douban/train_model_weights_epochs_douban_movie_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #         'best_models_02_23/douban/train_model_weights_epochs_douban_music_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/'], metavar='N',
    #                     help='the best source path')
    # parser.add_argument('--save_model_path', type=list, default=[
    #     'best_models/amazon/amazon_elec_cloth_phone/elec_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model',
    #     'best_models/amazon/amazon_elec_cloth_phone/cloth_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model',
    # 'best_models/amazon/amazon_elec_cloth_phone/phone_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model'],
    # metavar='N', help='the bets model path')
    # parser.add_argument('--save_model_path', type=list, default=[
    #     'best_models/amazon/amazon_phone_sport/phone_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #     'best_models/amazon/amazon_phone_sport/sport_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model'],
    #                     metavar='N', help='the bets model path')
    # parser.add_argument('--train_model_weights_path', type=list,
    # default=['best_models_02_23/amazon/amazon_phone_sport/train_model_weights_epochs_amazon_phone_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    # 'best_models_02_23/amazon/amazon_phone_sport/train_model_weights_epochs_amazon_sport_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #                                                  ], metavar='N',
    #                                         help='the best source path')


    args = parser.parse_args()
    return args