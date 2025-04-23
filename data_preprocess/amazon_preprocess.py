import pandas as pd
from loguru import logger

def filter_g_k_one(data,k=10,u_name='user_id',i_name='item_id',y_name='rating'):
    '''
    delete the records that user and item interactions lower than k
    '''
    item_group = data.groupby(i_name).agg({y_name:'count'}) #every item has the number of ratings
    item_g10 = item_group[item_group[y_name]>=k].index
    data_new = data[data[i_name].isin(item_g10)]

    user_group = data_new.groupby(u_name).agg({y_name: 'count'})  # every item has the number of ratings
    user_g10 = user_group[user_group[y_name] >= k].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new
def id_map(df,overlap_u_id_map_dict,nonoverlap_u_id_map,user_to_path,item_to_path,to_path):
    df_items = df.sort_values(by=['item_id'])
    uni_items = df_items['item_id'].unique().tolist()
    i_id_map = {k: i for i, k in enumerate(uni_items)}
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['item_id', 'itemID'])
    df['itemID'] = df['item_id'].map(i_id_map)
    df['itemID'] = df['itemID'].astype(int)

    overlap_u_df = pd.DataFrame(list(overlap_u_id_map_dict.items()), columns=['user_id', 'userID'])
    non_overlap_u_df = pd.DataFrame(list(nonoverlap_u_id_map.items()), columns=['user_id', 'userID'])
    u_df = pd.concat([overlap_u_df,non_overlap_u_df],axis=0)
    u_id_map = {**overlap_u_id_map_dict,**nonoverlap_u_id_map}
    df['userID'] = df['user_id'].map(u_id_map)
    df['userID'] = df['userID'].astype(int)

    u_df.to_csv(user_to_path,index=False)
    i_df.to_csv(item_to_path,index=False)
    df.to_csv(to_path,index=False)

def overlap_user_id_map(intersection_users):
    intersection_users = sorted(intersection_users)
    u_id_map = {k: i for i, k in enumerate(intersection_users)}
    return u_id_map

def non_overlap_user_id_map(intersection_users,df):
    df_filter = df[~df['user_id'].isin(intersection_users)]
    inter_max_uid = len(intersection_users)

    df_users = df_filter.sort_values(by=['user_id'])
    uni_users = df_users['user_id'].unique().tolist()
    u_id_map = {k: i+inter_max_uid for i, k in enumerate(uni_users)}

    return u_id_map


if __name__ == '__main__':
    # ------------process amazon cloth data
    # df_source = pd.read_csv('/data/lwang9/datasets/amazon/ratings/ratings_Cell_Phones_and_Accessories.csv')
    df_source = pd.read_csv('/data/lwang9/datasets/amazon/ratings/ratings_Cell_Phones_and_Accessories.csv')
    df_source.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df_source = df_source.drop_duplicates(keep='first')
    # filter user and item that have interactions less than 5
    df_source_new = filter_g_k_one(data=df_source,k=10,u_name='user_id',i_name='item_id',y_name='rating')
    source_users = list(df_source_new['user_id'].unique())
    logger.info('source total samples:{}, user num:{}, item num:{}, density:{}',len(df_source_new),len(source_users),
                len(df_source_new['item_id'].unique()),len(df_source_new)/(len(source_users)*len(df_source_new['item_id'].unique())))

    df_target = pd.read_csv('/data/lwang9/datasets/amazon/ratings/ratings_Sports_and_Outdoors.csv')
    df_target.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df_target = df_target.drop_duplicates(keep='first')
    # df_movie.dropna(inplace=True)
    df_target_new = filter_g_k_one(data=df_target,k=10,u_name='user_id',i_name='item_id',y_name='rating')
    target_users = list(df_target_new['user_id'].unique())
    logger.info('target total samples:{}, user num:{}, item num:{}, density:{}', len(df_target_new), len(target_users),
                len(df_target_new['item_id'].unique()),len(df_target_new)/(len(target_users)*len(df_target_new['item_id'].unique())))

    # df_target_1 = pd.read_csv('/data/lwang9/datasets/amazon/ratings/ratings_Cell_Phones_and_Accessories.csv')
    # df_target_1.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # df_target_1 = df_target_1.drop_duplicates(keep='first')
    # # df_movie.dropna(inplace=True)
    # df_target_1_new = filter_g_k_one(data=df_target_1, k=10, u_name='user_id', i_name='item_id', y_name='rating')
    # target_users_1 = list(df_target_1_new['user_id'].unique())
    # logger.info('target total samples:{}, user num:{}, item num:{}, density:{}', len(df_target_1_new), len(target_users_1),
    #             len(df_target_1_new['item_id'].unique()),
    #             len(df_target_1_new) / (len(target_users_1) * len(df_target_1_new['item_id'].unique())))

    #get common users
    intersection_users = list(set(source_users).intersection(target_users))
    # intersection_users = list(set(source_users).intersection(target_users,target_users_1))
    logger.info('common user num:{}',len(intersection_users))
    overlap_user_id_map_dict = overlap_user_id_map(intersection_users)
    source_nonoverlap_u_id_map= non_overlap_user_id_map(intersection_users=intersection_users,df=df_source_new)
    target_nonoverlap_u_id_map = non_overlap_user_id_map(intersection_users=intersection_users,df=df_target_new)
    # target_1_nonoverlap_u_id_map = non_overlap_user_id_map(intersection_users=intersection_users, df=df_target_1_new)

    id_map(df=df_source_new,overlap_u_id_map_dict=overlap_user_id_map_dict,
           nonoverlap_u_id_map=source_nonoverlap_u_id_map,
           user_to_path='../datasets/amazon_phone_sport/amazon_phone/user_id_map.csv',
           item_to_path='../datasets/amazon_phone_sport/amazon_phone/item_id_map.csv',
           to_path='../datasets/amazon_phone_sport/amazon_phone/phone_data.csv')
    #
    id_map(df=df_target_new, overlap_u_id_map_dict=overlap_user_id_map_dict,
           nonoverlap_u_id_map=target_nonoverlap_u_id_map,
           user_to_path='../datasets/amazon_phone_sport/amazon_sport/user_id_map.csv',
           item_to_path='../datasets/amazon_phone_sport/amazon_sport/item_id_map.csv',
           to_path='../datasets/amazon_phone_sport/amazon_sport/sport_data.csv')

    # id_map(df=df_target_1_new, overlap_u_id_map_dict=overlap_user_id_map_dict,
    #        nonoverlap_u_id_map=target_1_nonoverlap_u_id_map,
    #        user_to_path='../datasets/amazon_elec_cloth_phone/amazon_phone/user_id_map.csv',
    #        item_to_path='../datasets/amazon_elec_cloth_phone/amazon_phone/item_id_map.csv',
    #        to_path='../datasets/amazon_elec_cloth_phone/amazon_phone/phone_data.csv')









