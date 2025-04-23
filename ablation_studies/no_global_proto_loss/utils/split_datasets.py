import pandas as pd
import random
from loguru import logger

def split_datasets(df_rating):
    '''
        split train data and test data
        :param df:
        :return:
        '''
    train = []
    test = []
    for x in df_rating.groupby(by='userID'):
        test_item = random.choice(list(x[1]['itemID']))
        each_test = x[1][x[1]['itemID'].isin([test_item])][['userID', 'itemID', 'rating']]
        test.append(each_test)
        items = list(x[1]['itemID'])
        train_items = list(set(items).difference(set([test_item])))
        each_train = x[1][x[1]['itemID'].isin(train_items)][['userID', 'itemID', 'rating']]
        train.append(each_train)

    train_df = pd.concat(train, axis=0, ignore_index=True)
    # self.train_df.to_csv(self.train_path, index=False)
    logger.info('the number of train data is {}'.format(len(train_df)))

    test_df = pd.concat(test, axis=0, ignore_index=True)
    # self.test_df.to_csv(self.test_path, index=False)
    logger.info('the number of test data is {}'.format(len(test_df)))
    return train_df,test_df

if __name__ == '__main__':
    field = 'movie_music'
    dataset = 'music'
    path = f'/data/lwang9/CDR_data_process/douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/{dataset}_review_data_new.csv'
    df = pd.read_csv(path)
    train_df,test_df = split_datasets(df)
    train_to_path = f'../datasets/douban/{field}/{dataset}/train.csv'
    test_to_path = f'../datasets/douban/{field}/{dataset}/test.csv'
    train_df.to_csv(train_to_path,index=False)
    test_df.to_csv(test_to_path,index=False)