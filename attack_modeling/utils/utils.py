import numpy as np

def calculate_cos_sim(vec_1,vec_2):
    cosine_sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*(np.linalg.norm(vec_2)))
    return cosine_sim

