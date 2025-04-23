# import os
# import sys
# curPath = os.path.abspath(os.path.dirname((__file__)))
# rootPath = os.path.split(curPath)[0]
# PathProject = os.path.split(rootPath)[0]
# sys.path.append(rootPath)
# sys.path.append(PathProject)
import os

import os
import sys
curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import pandas as pd
import pickle
import random
import torch
import numpy as np
from loguru import logger

# from para_parser import parse
# args = parse()


class Load_Data(object):

    def __init__(self, **params):
        self.rating_file_path = params['rating_path']
        self.args = params['args']
        self.user_rev_emb_path = params['user_rev_emb_path']
        self.item_rev_emb_path = params['item_rev_emb_path']
        self.train_path = params['train_path']
        self.test_path = params['test_path']
        self.test_neg_path = params['test_neg_path']
        self.overlap_user_num = params['overlap_user_num']
        self.overlap_users = [i for i in range(self.overlap_user_num)]
        logger.info('load datasets')
        self.load_orig_dataset()
        # self.split_datasets()
        logger.info('create data loader')
        self.create_data_loader()
        #modify version, rduce test time
        logger.info('read test samples')
        self.read_test_samples()

    def load_orig_dataset(self):
        self.df_rating = pd.read_csv(self.rating_file_path)[['userID','itemID','rating']]
        self.df_rating['category'] = 0
        self.num_users = len(self.df_rating['userID'].unique())
        self.num_items = len(self.df_rating['itemID'].unique())
        self.train_df = pd.read_csv(self.train_path)
        self.train_df['category'] = 0
        self.test_df = pd.read_csv(self.test_path)
        self.test_df['category'] = 0
        # self.df_rating['centroid'] = [np.array([0]) for i in range(len(self.df_rating))]
        with open(self.user_rev_emb_path,'rb') as f_user:
            self.user_review_emb = torch.from_numpy(np.load(f_user))
        with open(self.item_rev_emb_path,'rb') as f_item:
            self.item_review_emb = torch.from_numpy(np.load(f_item))

    def split_datasets(self):
        '''
            split train data and test data
            :param df:
            :return:
            '''
        train = []
        test = []
        for x in self.df_rating.groupby(by='userID'):
            test_item = random.choice(list(x[1]['itemID']))
            each_test = x[1][x[1]['itemID'].isin([test_item])][['userID', 'itemID', 'category','rating']]
            test.append(each_test)
            items = list(x[1]['itemID'])
            train_items = list(set(items).difference(set([test_item])))
            each_train = x[1][x[1]['itemID'].isin(train_items)][['userID', 'itemID', 'category', 'rating']]
            train.append(each_train)

        self.train_df = pd.concat(train, axis=0, ignore_index=True)
        # self.train_df.to_csv(self.train_path, index=False)
        logger.info('the number of train data is {}'.format(len(self.train_df)))

        self.test_df = pd.concat(test, axis=0, ignore_index=True)
        # self.test_df.to_csv(self.test_path, index=False)
        logger.info('the number of test data is {}'.format(len(self.test_df)))

    def create_data_loader(self):
        train_u, train_v, train_c,train_r = self.train_df['userID'].values.tolist(), self.train_df['itemID'].values.tolist(), \
                                    self.train_df['category'].values.tolist(),self.train_df['rating'].values.tolist()
        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),torch.FloatTensor(train_c),
                                                  torch.FloatTensor(train_r))
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    def generate_train_instances(self,batch_u,batch_v,batch_category):
        num_items = len(self.df_rating['itemID'].unique())
        uids,iids,categories,ratings = [],[],[],[]
        for i in range(len(batch_u)):
            u_id = batch_u[i]
            pos_i_id = batch_v[i]
            uids.append(u_id)
            iids.append(pos_i_id)
            categories.append(batch_category[i])
            ratings.append(1)
            for t in range(self.args.train_neg_num):
                j = np.random.randint(num_items)
                while len(self.df_rating[(self.df_rating['userID'] == u_id) & (self.df_rating['itemID'] == j)]) > 0:
                    j = np.random.randint(num_items)
                uids.append(u_id)
                iids.append(j)
                categories.append(batch_category[i])
                ratings.append(0)
        return uids,iids,categories,ratings


    def read_test_samples(self):
        self.test_pos_samples = []
        self.test_neg_samples = []
        with open(self.test_neg_path) as f:
            for line in f:
                # each_neg_samples = []
                line = line.replace('\n','')
                each_content = line.split(' ')[:100]
                each_content = list(map(int,each_content))
                self.test_pos_samples.append([each_content[0],each_content[1]])
                self.test_neg_samples.append(each_content[2:100])

    def update_cat(self,category_dict):
        for key,value in category_dict.items():
            self.train_df.loc[self.train_df['userID'] == int(key), 'category'] = value
            self.test_df.loc[self.test_df['userID'] == int(key), 'category'] = value
            # self.train_df.loc[self.train_df['userID'] == int(key), 'centroid'] = self.train_df.loc[self.train_df['userID'] == int(key), 'centroid'].apply(lambda x: x + np.array(value[1]))
        self.create_data_loader()

if __name__ == '__main__':

    # data_params = {
    #     'rating_path': '../datasets/debug_datasets/douban_book/book_review_data_new.csv',
    #     'user_rev_emb_path':'../datasets/debug_datasets/douban_book/doc_embs/book_user_doc_emb_32.npy',
    #     'item_rev_emb_path':'../datasets/debug_datasets/douban_book/doc_embs/book_item_doc_emb_32.npy',
    #     'args':args,
    #     'overlap_user_num':6
    # }
    # df = pd.read_csv('/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth/cloth_data.csv')
    # print(len(df))
    data_params = {
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth/cloth_data.csv',
        'args':args,
        'overlap_user_num': 1284
    }
    load_data = Load_Data(**data_params)
    print('gg')