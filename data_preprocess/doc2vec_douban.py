import pickle
from loguru import logger
import numpy as np

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # import stopwords corpus
import string  # delete various punctuation


# download stopwords
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


def tokenize_text(text):
    # print(text)
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

def read_stop_words():
    with open('../douban_data_process_FedPCL_MDR/datasets/cn_stopwords.txt', 'r') as file:
        file_content = file.read().split('\n')
    return set(file_content)

def doc_model_train(df,model_save_path,emb_save_path,tag_name,vector_size):
    # stop_words = read_stop_words()
    # generate TaggedDocument object for each user
    tagged_data = []
    logger.info('process documents')
    for index, row in df.iterrows():
        documents = [TaggedDocument(words=tokenize_text(row['reviews']), tags=[row[tag_name]])]
        tagged_data.extend(documents)

    # train Doc2Vec model
    logger.info('train doc2vec model')
    model = Doc2Vec(vector_size=vector_size, window=5, min_count=1, workers=4, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_save_path)
    # # obtain each user's Doc2Vec embedding
    logger.info('obtain user embeddings')
    user_embeddings = {}
    for index, row in df.iterrows():
        user_id = row[tag_name]
        user_embedding = model.dv[user_id]
        user_embeddings[user_id] = user_embedding
    #
    # # save user_embeddings
    logger.info('save user embeddings')
    with open(emb_save_path, 'wb') as file:
        pickle.dump(user_embeddings, file)

logger.info('read stopwords')
stop_words = read_stop_words()
# stop_words = set(stopwords.words('english'))

def pickle_to_npy(source_path,to_path):
    all_vectors = []
    with open(source_path, 'rb') as f:
        text_feat = pickle.load(f)
    for key, value in text_feat.items():
        all_vectors.append(value)
    vectors_array = np.array(all_vectors)
    np.save(to_path, vectors_array)

if __name__ == '__main__':
    # read file
    logger.info('process user reviews')
    field = 'book_movie'
    dataset = 'book'
    column = 'item'
    column_name = 'itemID'
    file_path = f'../douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/{dataset}_{column}_reviews.csv'
    df = pd.read_csv(file_path)
    doc_model_train(df=df, model_save_path=f'../douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/doc_models'
                                           f'/{dataset}_{column}_doc_model_64.model',
        emb_save_path=f'../douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/doc_embs/{dataset}_{column}_doc_emb_64.pickle',
                    tag_name=column_name, vector_size=64)
    pickle_to_npy(source_path=f'../douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/doc_embs/{dataset}_{column}_doc_emb_64.pickle',
                  to_path=f'../douban_data_process_FedPCL_CDR/datasets/{field}/{dataset}/doc_embs/{dataset}_{column}_doc_emb_64.npy')
    #
    # file_path = '../douban_data_process_FedPCL_MDR/datasets/douban_movie/movie_user_reviews.csv'
    # df = pd.read_csv(file_path)
    # doc_model_train(df=df, model_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_models/movie_user_doc_model_64',
    #                 emb_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_user_doc_emb_64.pickle',
    #                 tag_name='userID', vector_size=64)
    #
    # file_path = '../douban_data_process_FedPCL_MDR/datasets/douban_music/music_user_reviews.csv'
    # df = pd.read_csv(file_path)
    # doc_model_train(df=df, model_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_models/music_user_doc_model_64',
    #                 emb_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_user_doc_emb_64.pickle',
    #                 tag_name='userID', vector_size=64)
    #
    # logger.info('process item reviews')
    # file_path = '../douban_data_process_FedPCL_MDR/datasets/douban_book/book_item_reviews.csv'
    # df = pd.read_csv(file_path)
    # doc_model_train(df=df,
    #                 model_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_models/book_item_doc_model_64.model',
    #                 emb_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_item_doc_emb_64.pickle',
    #                 tag_name='itemID', vector_size=64)
    #
    # file_path = '../douban_data_process_FedPCL_MDR/datasets/douban_movie/movie_item_reviews.csv'
    # df = pd.read_csv(file_path)
    # doc_model_train(df=df,
    #                 model_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_models/movie_item_doc_model_64',
    #                 emb_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_item_doc_emb_64.pickle',
    #                 tag_name='itemID', vector_size=64)
    #
    # file_path = '../douban_data_process_FedPCL_MDR/datasets/douban_music/music_item_reviews.csv'
    # df = pd.read_csv(file_path)
    # doc_model_train(df=df,
    #                 model_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_models/music_item_doc_model_64',
    #                 emb_save_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_item_doc_emb_64.pickle',
    #                 tag_name='itemID', vector_size=64)


    # #--------------------------------------
    # pickle_to_npy(source_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_64.pickle',
    #               to_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_64.npy')
    # pickle_to_npy(source_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_user_doc_emb_64.pickle',
    #               to_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_user_doc_emb_64.npy')
    # pickle_to_npy(
    #     source_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_user_doc_emb_64.pickle',
    #     to_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_user_doc_emb_64.npy')
    #
    # pickle_to_npy(
    #     source_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_item_doc_emb_64.pickle',
    #     to_path='../douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_item_doc_emb_64.npy')
    # pickle_to_npy(
    #     source_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_item_doc_emb_64.pickle',
    #     to_path='../douban_data_process_FedPCL_MDR/datasets/douban_movie/doc_embs/movie_item_doc_emb_64.npy')
    # pickle_to_npy(
    #     source_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_item_doc_emb_64.pickle',
    #     to_path='../douban_data_process_FedPCL_MDR/datasets/douban_music/doc_embs/music_item_doc_emb_64.npy')




