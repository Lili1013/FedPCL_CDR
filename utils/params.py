
data_params = {
    # 'douban_book':{
    #     'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/book_review_data_new.csv',
    #     'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_32.npy',
    #     'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_item_doc_emb_32.npy',
    #     'train_path': 'datasets/douban/book/train.csv',
    #     'test_path': 'datasets/douban/book/test.csv',
    #     'test_neg_path':'datasets/douban/book/test.txt',
    #     'overlap_user_num': 1008
    # },
    # 'douban_movie':{
    #     'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/movie_review_data_new.csv',
    #     'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_user_doc_emb_32.npy',
    #     'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_item_doc_emb_32.npy',
    #     'overlap_user_num': 1008,
    #     'train_path': 'datasets/douban/movie/train.csv',
    #     'test_path': 'datasets/douban/movie/test.csv',
    #     'test_neg_path':'datasets/douban/movie/test.txt',
    # },
    # 'douban_music':{
    #     'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/music_review_data_new.csv',
    #     'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_user_doc_emb_32.npy',
    #     'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_item_doc_emb_32.npy',
    #     'overlap_user_num': 1008,
    #     'train_path': 'datasets/douban/music/train.csv',
    #     'test_path': 'datasets/douban/music/test.csv',
    #     'test_neg_path':'datasets/douban/music/test.txt',
    # },
    'douban_book':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/book_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_item_doc_emb_64.npy',
        'train_path': '../datasets/douban/book/train.csv',
        'test_path': '../datasets/douban/book/test.csv',
        'test_neg_path':'../datasets/douban/book/test.txt',
        'overlap_user_num': 1008
    },
    'douban_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/movie_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_item_doc_emb_64.npy',
        'overlap_user_num': 1008,
        'train_path': '../datasets/douban/movie/train.csv',
        'test_path': '../datasets/douban/movie/test.csv',
        'test_neg_path':'../datasets/douban/movie/test.txt',
    },
    'douban_music':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/music_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_item_doc_emb_64.npy',
        'overlap_user_num': 1008,
        'train_path': '../datasets/douban/music/train.csv',
        'test_path': '../datasets/douban/music/test.csv',
        'test_neg_path':'../datasets/douban/music/test.txt',
    },
    'douban_book_movie_book':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/book/book_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/book/doc_embs/book_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/book/doc_embs/book_item_doc_emb_64.npy',
        'train_path': '../datasets/douban/book_movie/book/train.csv',
        'test_path': '../datasets/douban/book_movie/book/test.csv',
        'test_neg_path':'../datasets/douban/book_movie/book/test.txt',
        'overlap_user_num': 1585
    },
    'douban_book_movie_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/movie/movie_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/movie/doc_embs/movie_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/book_movie/movie/doc_embs/movie_item_doc_emb_64.npy',
        'train_path': 'datasets/douban/book_movie/movie/train.csv',
        'test_path': 'datasets/douban/book_movie/movie/test.csv',
        'test_neg_path':'datasets/douban/book_movie/movie/test.txt',
        'overlap_user_num': 1585
    },
    'douban_movie_music_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/movie/movie_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/movie/doc_embs/movie_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/movie/doc_embs/movie_item_doc_emb_64.npy',
        'train_path': '../datasets/douban/movie_music/movie/train.csv',
        'test_path': '../datasets/douban/movie_music/movie/test.csv',
        'test_neg_path':'../datasets/douban/movie_music/movie/test.txt',
        'overlap_user_num': 1134
    },
    'douban_movie_music_music':{
        'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/music/music_review_data_new.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/music/doc_embs/music_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/movie_music/music/doc_embs/music_item_doc_emb_64.npy',
        'train_path': '../datasets/douban/movie_music/music/train.csv',
        'test_path': '../datasets/douban/movie_music/music/test.csv',
        'test_neg_path':'../datasets/douban/movie_music/music/test.txt',
        'overlap_user_num': 1134
    },
    'amazon_cloth_sport_cloth':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth_sport/amazon_cloth/cloth_data.csv',
        'train_path':'datasets/amazon/amazon_cloth_sport/amazon_cloth/train.csv',
        'test_path':'datasets/amazon/amazon_cloth_sport/amazon_cloth/test.csv',
        'test_neg_path':'datasets/amazon/amazon_cloth_sport/amazon_cloth/test.txt',
        'overlap_user_num': 1284
    },
    'amazon_cloth_sport_sport':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth_sport/amazon_sport/sport_data.csv',
        'overlap_user_num': 1284,
        'train_path':'datasets/amazon/amazon_cloth_sport/amazon_sport/train.csv',
        'test_path':'datasets/amazon/amazon_cloth_sport/amazon_sport/test.csv',
        'test_neg_path':'datasets/amazon/amazon_cloth_sport/amazon_sport/test.txt',
    },
    'amazon_phone_sport_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_phone/phone_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_phone/doc_embs/phone_user_doc_model_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_phone/doc_embs/phone_item_doc_model_64.npy',
        'overlap_user_num': 655,
        'train_path': '../datasets/amazon/amazon_phone_sport/amazon_phone/train.csv',
        'test_path': '../datasets/amazon/amazon_phone_sport/amazon_phone/test.csv',
        'test_neg_path': '../datasets/amazon/amazon_phone_sport/amazon_phone/test.txt',
    },
    'amazon_phone_sport_sport':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_sport/sport_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_sport/doc_embs/sport_user_doc_model_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_sport/doc_embs/sport_item_doc_model_64.npy',
        'overlap_user_num': 655,
        'train_path': '../datasets/amazon/amazon_phone_sport/amazon_sport/train.csv',
        'test_path': '../datasets/amazon/amazon_phone_sport/amazon_sport/test.csv',
        'test_neg_path': '../datasets/amazon/amazon_phone_sport/amazon_sport/test.txt',
    },

    'amazon_elec_cloth_phone_elec':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_elec/elec_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_elec/doc_embs/elec_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_elec/doc_embs/elec_item_doc_emb_64.npy',
        'overlap_user_num': 468,
        'train_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_elec/train.csv',
        'test_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_elec/test.csv',
        'test_neg_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_elec/test.txt',
    },
    'amazon_elec_cloth_phone_cloth':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_cloth/cloth_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_cloth/doc_embs/cloth_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_cloth/doc_embs/cloth_item_doc_emb_64.npy',
        'overlap_user_num': 468,
        'train_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_cloth/train.csv',
        'test_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_cloth/test.csv',
        'test_neg_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_cloth/test.txt',
    },
    'amazon_elec_cloth_phone_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_phone/phone_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_phone/doc_embs/phone_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_elec_cloth_phone/amazon_phone/doc_embs/phone_item_doc_emb_64.npy',
        'overlap_user_num': 468,
        'train_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_phone/train.csv',
        'test_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_phone/test.csv',
        'test_neg_path': '../datasets/amazon/amazon_elec_cloth_phone/amazon_phone/test.txt',
    },
    'amazon_phone_elec_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_phone/phone_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_phone/doc_embs/phone_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_phone/doc_embs/phone_item_doc_emb_64.npy',
        'overlap_user_num': 2904,
        'train_path': 'datasets/amazon/amazon_phone_elec/amazon_phone/train.csv',
        'test_path': 'datasets/amazon/amazon_phone_elec/amazon_phone/test.csv',
        'test_neg_path': 'datasets/amazon/amazon_phone_elec/amazon_phone/test.txt',
    },
    'amazon_phone_elec_elec':{
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_elec/elec_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_elec/doc_embs/elec_user_doc_emb_64.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_elec/doc_embs/elec_item_doc_emb_64.npy',
        'overlap_user_num': 2904,
        'train_path': 'datasets/amazon/amazon_phone_elec/amazon_elec/train.csv',
        'test_path': 'datasets/amazon/amazon_phone_elec/amazon_elec/test.csv',
        'test_neg_path': 'datasets/amazon/amazon_phone_elec/amazon_elec/test.txt',
    }
}

model_params = {
    'douban_book':{
        'user_num' : 1715,
        'item_num' : 8660,
    },
    'douban_movie':{
        'user_num' : 2320,
        'item_num' : 5803,
    },
    'douban_music':{
        'user_num' : 1193,
        'item_num' : 7146,
    },
    'amazon_cloth':{
        'user_num' : 13058,
        'item_num' : 62137,
    },
    'amazon_sport':{
        'user_num':10849,
        'item_num':35368
    },
    'amazon_phone':{
        'user_num':5730,
        'item_num':22287
    },
    'amazon_elec': {
        'user_num': 12301,
        'item_num': 56081
    }

}
#
#
# data_params = {
#     'douban_book':{
#         'rating_path': 'datasets/debug_datasets/douban_book/book_review_data_new.csv',
#         'user_rev_emb_path': 'datasets/debug_datasets/douban_book/doc_embs/book_user_doc_emb_32.npy',
#         'item_rev_emb_path': 'datasets/debug_datasets/douban_book/doc_embs/book_item_doc_emb_32.npy',
#         # 'train_path': '../datasets/douban/douban_book/book_train_data.csv',
#         # 'test_path': '../datasets/douban/douban_book/book_test_data.csv',
#         'overlap_user_num': 6
#     },
#     'douban_movie':{
#         'rating_path': 'datasets/debug_datasets/douban_movie/movie_review_data_new.csv',
#         'user_rev_emb_path': 'datasets/debug_datasets/douban_movie/doc_embs/movie_user_doc_emb_32.npy',
#         'item_rev_emb_path': 'datasets/debug_datasets/douban_movie/doc_embs/movie_item_doc_emb_32.npy',
#         'overlap_user_num': 6
#     },
#     'douban_music':{
#         'rating_path': 'datasets/debug_datasets/douban_music/music_review_data_new.csv',
#         'user_rev_emb_path': 'datasets/debug_datasets/douban_music/doc_embs/music_user_doc_emb_32.npy',
#         'item_rev_emb_path': 'datasets/debug_datasets/douban_music/doc_embs/music_item_doc_emb_32.npy',
#         'overlap_user_num': 6
#     }
# }
#
# model_params = {
#     'douban_book':{
#         'user_num' : 78,
#         'item_num' : 8776,
#     },
#     'douban_movie':{
#         'user_num' : 13,
#         'item_num' : 6054,
#     },
#     'douban_music':{
#         'user_num' : 95,
#         'item_num' : 8664,
#     },
#
# }