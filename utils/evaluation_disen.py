import math
import heapq
import numpy as np
import torch
import heapq
from loguru import logger
import multiprocessing
import heapq
import random as rd


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    rd.seed(seed)


set_seed(2021)
cores = multiprocessing.cpu_count() // 2

_model = None
_test_ratings = None
_test_negatives = None
_k = None
_device = None


def calculate_hr_ndcg(model, test_ratings, test_negatives, k,device):
    global _model
    global _test_negatives
    global _test_ratings
    global _k
    global _device
    global _epoch
    global _test_user_records
    _model = model
    _test_ratings = test_ratings
    _test_negatives = test_negatives
    _k = k
    _device = device

    pool = multiprocessing.Pool(cores)
    source_hits, source_ndcgs, source_precisions, source_recalls = [], [], [], []
    target_hits, target_ndcgs, target_precisions, target_recalls = [], [], [], []
    test_user_num = len(_test_ratings)
    pred_ratings = np.zeros(shape=(test_user_num, (len(_test_negatives[0]) + 1)))


    test_records = np.zeros(shape=(test_user_num, (len(_test_negatives[0]) + 1)))


    # source_test_users = list(np.array(_source_test_ratings)[:,0])
    # target_test_users = list(np.array(_target_test_ratings)[:, 0])
    test_users = []
    test_pos_items_list = []


    for idx in range(len(_test_ratings)):
        u_id = _test_ratings[idx][0]
        test_users.append(u_id)
        test_pos_items = _test_ratings[idx][1:]
        test_pos_items_list.append(test_pos_items)
        test_neg_items = _test_negatives[idx]
        items = test_pos_items + test_neg_items
        test_records[idx, :] = items

        pred = pred_one_user(u_id, items)
        pred_ratings[idx, :] = pred.detach().cpu()

    #top_k=10----------------
    top_k_10 = [10 for i in range(len(test_users))]
    rating_uid = zip(pred_ratings, test_users, test_records, test_pos_items_list,top_k_10)

    result = pool.map(test_one_user, rating_uid)
    hr, ndcg, precision, recall, mrr = obtain_final_result(result)

    # top_k=1----------------
    top_k_2 = [2 for i in range(len(test_users))]
    source_rating_uid_2 = zip(pred_ratings, test_users, test_records, test_pos_items_list,
                            top_k_2)
    source_result_2 = pool.map(test_one_user, source_rating_uid_2)
    source_hr_2, source_ndcg_2, source_precision_2, source_recall_2, source_mrr_2 = obtain_final_result(source_result_2)


    # top_k=4----------------
    top_k_4 = [4 for i in range(len(test_users))]
    source_rating_uid_4 = zip(pred_ratings, test_users, test_records, test_pos_items_list,
                              top_k_4)
    source_result_4 = pool.map(test_one_user, source_rating_uid_4)
    source_hr_4, source_ndcg_4, source_precision_4, source_recall_4, source_mrr_4 = obtain_final_result(source_result_4)

    # top_k=6----------------
    top_k_6 = [6 for i in range(len(test_users))]
    source_rating_uid_6 = zip(pred_ratings, test_users, test_records, test_pos_items_list,
                              top_k_6)
    source_result_6 = pool.map(test_one_user, source_rating_uid_6)
    source_hr_6, source_ndcg_6, source_precision_6, source_recall_6, source_mrr_6 = obtain_final_result(source_result_6)

    # top_k=8----------------
    top_k_8 = [8 for i in range(len(test_users))]
    source_rating_uid_8 = zip(pred_ratings, test_users, test_records, test_pos_items_list,
                              top_k_8)
    source_result_8 = pool.map(test_one_user, source_rating_uid_8)
    source_hr_8, source_ndcg_8, source_precision_8, source_recall_8, source_mrr_8 = obtain_final_result(source_result_8)


    pool.close()

    return hr, ndcg, precision, recall, mrr, \
           source_hr_2, source_ndcg_2, source_precision_2, source_recall_2, source_mrr_2,\
           source_hr_4, source_ndcg_4, source_precision_4, source_recall_4, source_mrr_4,\
           source_hr_6, source_ndcg_6, source_precision_6, source_recall_6, source_mrr_6,\
           source_hr_8, source_ndcg_8, source_precision_8, source_recall_8, source_mrr_8


def obtain_final_result(source_result):
    source_result = np.array(source_result)
    source_hr = np.mean(source_result[:, 0])
    source_ndcg = np.mean(source_result[:, 1])
    source_precision = np.mean(source_result[:, 2])
    source_recall = np.mean(source_result[:, 3])
    source_mrr = np.mean(source_result[:, 4])
    return source_hr,source_ndcg,source_precision,source_recall,source_mrr


def test_one_user(x):
    rating = x[0]
    test_user_records = x[2]
    user_pos_test = x[3]
    top_k = x[4]
    test_items = test_user_records
    r = ranklist_by_heapq(test_items, rating,top_k)
    return eval_one_user(r, user_pos_test)


def ranklist_by_heapq(test_items, rating,top_k):
    item_score = {}
    for i in range(len(test_items)):
        item_score[int(test_items[i])] = rating[i]

    K_item_score = heapq.nlargest(top_k, item_score, key=item_score.get)
    return K_item_score


def pred_one_user(u_id, items):
    user_id_input, item_id_input = [], []

    for i in range(len(items)):
        user_id_input.append(u_id)
        item_id_input.append(items[i])

    user_id_input = torch.tensor(user_id_input).to(_device)
    item_id_input = torch.tensor(item_id_input).to(_device)

    u_c_feats,u_s_feats,v_feats, pred_prob = _model.forward(user_id_input, item_id_input)
    return pred_prob


def eval_one_user(r, user_pos_test):
    hr = get_hit_ratio(r, user_pos_test)
    precision = get_precision(r, user_pos_test)
    recall = get_recall(r, user_pos_test)
    ndcg = get_ndcg(r, user_pos_test)
    mrr = get_mrr(r, user_pos_test)
    # target_ndcg = get_ndcg(target_ranklist, target_i_id)
    # target_precision = get_precision(target_ranklist,_k)
    # target_recall = get_recall(target_ranklist, _k, 1)
    return [hr, ndcg, precision, recall, mrr]


def get_hit_ratio(ranklist, user_pos_test):
    for each_item in ranklist:
        if each_item in user_pos_test:
            return 1
    return 0


def get_ndcg(ranklist, user_pos_test):
    for i in range(len(ranklist)):
        each_item = ranklist[i]
        if each_item in user_pos_test:
            return math.log(2) / math.log(i + 2)
    return 0


def get_precision(ranklist, user_pos_test):
    precision_items = []
    for each in ranklist:
        if each in user_pos_test:
            precision_items.append(1)
    return sum(precision_items) / len(ranklist)


def get_recall(ranklist, user_pos_test):
    recall_items = []
    for each in ranklist:
        if each in user_pos_test:
            recall_items.append(1)
    return sum(recall_items) / len(user_pos_test)


def get_mrr(ranklist, user_pos_test):
    for i in range(len(ranklist)):
        if ranklist[i] in user_pos_test:
            return 1 / (i + 1)
    return 0


