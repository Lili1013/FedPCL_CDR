import pickle
from loguru import logger

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # import stopwords corpus
import string  # delete various punctuation
import numpy as np


# download stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def tokenize_text(text):
    # print(text)
    tokens = word_tokenize(str(text).lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

# def read_stop_words():
#     with open('datasets/stopwords-en.txt','r') as file:
#         file_content = file.read().split('\n')
#     return set(file_content)

# stop_words = read_stop_words()
stop_words = set(stopwords.words('english'))
if __name__ == '__main__':
    # read file
    path = '../amazon_data_process_FedPCL_MDR/datasets/amazon_phone_elec/amazon_phone/'
    dataset = 'phone'
    name = 'user'
    column_name = 'userID'
    rev_path = f'{path}{dataset}_{name}_reviews.csv'
    model_save_path = f'{path}doc_models/{dataset}_{name}_doc_model_64.model'
    emb_save_path = f'{path}doc_embs/{dataset}_{name}_doc_emb_64.pickle'
    npy_emb_save_path = f'{path}doc_embs/{dataset}_{name}_doc_emb_64.npy'
    logger.info('read file')
    df = pd.read_csv(rev_path)
    # tokennize and delete stopwords
    logger.info('download stopwords')

    # stop_words = read_stop_words()
    # generate TaggedDocument object for each user
    tagged_data = []
    logger.info('process documents')
    for index, row in df.iterrows():
        documents = [TaggedDocument(words=tokenize_text(row['review']), tags=[row[column_name]])]
        tagged_data.extend(documents)

    # train Doc2Vec model
    logger.info('train doc2vec model')
    model = Doc2Vec(vector_size=64, window=5, min_count=1, workers=4, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_save_path)
    # # obtain each user's Doc2Vec embedding
    logger.info('obtain user embeddings')
    user_embeddings = {}
    for index, row in df.iterrows():
        user_id = row[column_name]
        user_embedding = model.dv[user_id]
        user_embeddings[user_id] = user_embedding
    #
    # # save user_embeddings
    logger.info('save user embeddings')
    with open(emb_save_path,'wb') as file:
        pickle.dump(user_embeddings,file)

    all_vectors = []
    with open(emb_save_path, 'rb') as f:
        text_feat = pickle.load(f)
    for key, value in text_feat.items():
        all_vectors.append(value)
    vectors_array = np.array(all_vectors)
    np.save(npy_emb_save_path, vectors_array)
