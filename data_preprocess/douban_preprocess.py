import pandas as pd

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
    # ------------process douban bood review data
    df_book = pd.read_csv('../datasets/douban_book/book_review_data.csv')
    df_book = df_book.drop_duplicates(keep='first')
    # filter user and item that have interactions less than 5
    df_book_new = filter_g_k_one(data=df_book,k=5,u_name='user_id',i_name='item_id',y_name='rating')
    book_users = list(df_book_new['user_id'].unique())

    df_movie = pd.read_csv('../datasets/douban_movie/movie_review_data.csv')
    # df_movie = df_movie.sample(frac=0.1,random_state=42)
    df_movie = df_movie.drop_duplicates(keep='first')
    # df_movie.dropna(inplace=True)
    df_movie_new = filter_g_k_one(data=df_movie,k=5,u_name='user_id',i_name='item_id',y_name='rating')
    df_movie_new['user_id'] = df_movie_new['user_id'].astype(int)
    df_movie_new['item_id'] = df_movie_new['item_id'].astype(int)
    df_movie_new['rating'] = df_movie_new['rating'].astype(float)
    movie_users = list(df_movie_new['user_id'].unique())
    #
    df_music = pd.read_csv('../datasets/douban_music/music_review_data.csv')
    df_music = df_music.drop_duplicates(keep='first')
    df_music_new = filter_g_k_one(data=df_music,k=5,u_name='user_id',i_name='item_id',y_name='rating')
    music_users = list(df_music_new['user_id'].unique())

    #get common users
    intersection_users = list(set(book_users).intersection(movie_users,music_users))
    print(len(intersection_users))
    overlap_user_id_map_dict = overlap_user_id_map(intersection_users)
    book_nonoverlap_u_id_map= non_overlap_user_id_map(intersection_users=intersection_users,df=df_book_new)
    movie_nonoverlap_u_id_map = non_overlap_user_id_map(intersection_users=intersection_users,df=df_movie_new)
    music_nonoverlap_u_id_map= non_overlap_user_id_map(intersection_users=intersection_users,df=df_music_new)
    id_map(df=df_book_new,overlap_u_id_map_dict=overlap_user_id_map_dict,
           nonoverlap_u_id_map=book_nonoverlap_u_id_map,
           user_to_path='../datasets/douban_book/book_user_id_map.csv',
           item_to_path='../datasets/douban_book/book_item_id_map.csv',
           to_path='../datasets/douban_book/book_review_data_new.csv')
    #
    id_map(df=df_movie_new, overlap_u_id_map_dict=overlap_user_id_map_dict,
           nonoverlap_u_id_map=movie_nonoverlap_u_id_map,
           user_to_path='../datasets/douban_movie/movie_user_id_map.csv',
           item_to_path='../datasets/douban_movie/movie_item_id_map.csv',
           to_path='../datasets/douban_movie/movie_review_data_new.csv')

    id_map(df=df_music_new, overlap_u_id_map_dict=overlap_user_id_map_dict,
           nonoverlap_u_id_map=music_nonoverlap_u_id_map,
           user_to_path='../datasets/douban_music/music_user_id_map.csv',
           item_to_path='../datasets/douban_music/music_item_id_map.csv',
           to_path='../datasets/douban_music/music_review_data_new.csv')

    # ----------concat reviews for each user and item
    df_book = pd.read_csv('../datasets/douban_book/book_review_data_new.csv')
    book_group_df = df_book.groupby('userID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    book_group_df.to_csv('../datasets/douban_book/book_user_reviews.csv', index=False)
    book_group_df = df_book.groupby('itemID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    book_group_df.to_csv('../datasets/douban_book/book_item_reviews.csv', index=False)

    df_movie = pd.read_csv('../datasets/douban_movie/movie_review_data_new.csv')
    df_movie['reviews'] = df_movie['reviews'].astype(str)
    movie_group_df = df_movie.groupby('userID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    movie_group_df.to_csv('../datasets/douban_movie/movie_user_reviews.csv', index=False)
    movie_group_df = df_movie.groupby('itemID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    movie_group_df.to_csv('../datasets/douban_movie/movie_item_reviews.csv', index=False)

    df_music = pd.read_csv('../datasets/douban_music/music_review_data_new.csv')
    music_group_df = df_music.groupby('userID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    music_group_df.to_csv('../datasets/douban_music/music_user_reviews.csv', index=False)
    music_group_df = df_music.groupby('itemID')['reviews'].agg(lambda x: ' '.join(x)).reset_index()
    music_group_df.to_csv('../datasets/douban_music/music_item_reviews.csv', index=False)








