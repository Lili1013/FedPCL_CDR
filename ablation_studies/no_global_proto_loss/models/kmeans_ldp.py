import torch
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.mixture import GaussianMixture

def add_laplace_noise(data,noise_control):
    # self.beta = self.sensitivity/self.epsilon
    noise = np.random.laplace(0, noise_control, data.shape)
    # noise = np.random.normal(loc=0,scale=noise_control,size=data.shape)
    # noise = torch.tensor(np.random.normal(0, noise_control, data.shape))
    # noise = noise.to(torch.float32)
    # noisy_data = data+noise
    return noise

def gmm_clustering(num_clusters, embeddings, overlap_users):
    # Initialize GMM model
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)

    # Fit the GMM model on the embeddings
    gmm.fit(embeddings)

    # Get the cluster labels and means (centroids) of the GMM
    cluster_labels = gmm.predict(embeddings)
    centroids = gmm.means_  # centroids of the GMM clusters

    # Count the number of samples per cluster
    label_count_dict = Counter(cluster_labels)

    # Initialize dictionaries for storing results
    label_dict = {}
    cluster_overlap_smaples = {}

    # Populate the label_dict with sample ID and corresponding cluster label
    for id, label in enumerate(cluster_labels):
        label_dict[id] = label
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        if cluster_label not in cluster_overlap_smaples:
            cluster_overlap_smaples[cluster_label] = [[],[],[label_count_dict[cluster_label]]]
        if i in overlap_users:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
            cluster_overlap_smaples[cluster_label][1].append(i)
        else:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)

    # Iterate through all the embeddings and assign them to the appropriate cluster
    # for i in range(len(embeddings)):
    #     cluster_label = cluster_labels[i]
    #     centroid = cluster_means[cluster_label]
    # 
    #     # Initialize the cluster dictionary if it doesn't exist
    #     if cluster_label not in cluster_overlap_samples:
    #         cluster_overlap_samples[cluster_label] = [[], [], [label_count_dict[cluster_label]]]
    # 
    #     # Convert centroid to a tuple for comparison (arrays cannot be directly compared to lists)
    #     centroid_tuple = tuple(centroid)
    # 
    #     # If the user is in the overlap_users set, extend the centroid and add the sample index
    #     if i in overlap_users:
    #         # Check if the centroid is not already in the list
    #         if not any(np.array_equal(np.array(centroid_tuple), np.array(existing_centroid)) for existing_centroid in
    #                    cluster_overlap_samples[cluster_label][0]):
    #             cluster_overlap_samples[cluster_label][0].append(centroid_tuple)
    #         cluster_overlap_samples[cluster_label][1].append(i)
    #     else:
    #         # If the sample is not in overlap_users, just add the centroid without the sample index
    #         if not any(np.array_equal(np.array(centroid_tuple), np.array(existing_centroid)) for existing_centroid in
    #                    cluster_overlap_samples[cluster_label][0]):
    #             cluster_overlap_samples[cluster_label][0].append(centroid_tuple)

    # Return the cluster assignments, the overlap samples, and the unique cluster labels
    return label_dict, cluster_overlap_smaples, set(cluster_labels)


def k_means(num_clusters,embeddings,overlap_users,args):
    kmeans = KMeans(n_clusters=num_clusters,random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    for id,label in enumerate(cluster_labels):
        # centroid = centroids[label]
        label_dict[id] = label
    cluster_overlap_smaples = {}
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        # noise = add_laplace_noise(centroid, 0.2)
        # centroid = centroid + noise
        if cluster_label not in cluster_overlap_smaples:
            cluster_overlap_smaples[cluster_label] = [[],[],[],[label_count_dict[cluster_label]]]
        if i in overlap_users:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
                l2_norm = np.linalg.norm(centroid)  # Compute L2 norm using NumPy
                clip_centroid_feat = centroid / max(1, l2_norm / args.C)
                clip_centroid_feat = clip_centroid_feat
                noise_centroid = clip_centroid_feat+add_laplace_noise(clip_centroid_feat,args.lap_noise)
                cluster_overlap_smaples[cluster_label][1].extend(noise_centroid)
            cluster_overlap_smaples[cluster_label][2].append(i)
        else:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
                l2_norm = np.linalg.norm(centroid)  # Compute L2 norm using NumPy
                clip_centroid_feat = centroid / max(1, l2_norm / args.C)
                clip_centroid_feat = clip_centroid_feat
                noise_centroid = clip_centroid_feat + add_laplace_noise(clip_centroid_feat, args.lap_noise)
                cluster_overlap_smaples[cluster_label][1].extend(noise_centroid)

    return label_dict,cluster_overlap_smaples,set(cluster_labels)

def select_K(max_clusters,embeddings):
    sse = []
    for k in range(2,max_clusters):
        k_means = KMeans(n_clusters=k,random_state=42)
        k_means.fit(embeddings)
        sse.append(silhouette_score(embeddings,k_means.labels_))
    plt.plot(range(2,max_clusters),sse)
    plt.show()

def kmeans_temp(clusters,embeddings):
    k_means = KMeans(n_clusters=clusters, random_state=42)
    k_means.fit(embeddings)

    centroids = k_means.cluster_centers_
    cluster_labels = k_means.labels_
    return centroids,cluster_labels


def draw(embeddings,labels,centers):
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels,cmap='viridis',edgecolors='k',s=50)
    plt.scatter(centers[:,0],centers[:,1],c='red',marker='X',s=200,label='Cluster Centers')
    plt.savefig('jj.pdf')
    plt.show()

def spectral_cluster(num_clusters,embeddings,overlap_users):
    spectral = SpectralClustering(n_clusters=num_clusters,affinity='nearest_neighbors',random_state=42)
    cluster_labels = spectral.fit_predict(embeddings)
    centroids = []
    for cluster_label in np.unique(cluster_labels):
        cluster_points = embeddings[cluster_labels==cluster_label]
        cluster_center = np.mean(cluster_points,axis=0)
        centroids.append(cluster_center)
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    for id, label in enumerate(cluster_labels):
        # centroid = centroids[label]
        label_dict[id] = label
    cluster_overlap_smaples = {}
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        if cluster_label not in cluster_overlap_smaples:
            cluster_overlap_smaples[cluster_label] = [[],[],[label_count_dict[cluster_label]]]
        if i in overlap_users:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
            cluster_overlap_smaples[cluster_label][1].append(i)
        else:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
    return label_dict,cluster_overlap_smaples,set(cluster_labels)



if __name__ == '__main__':
    # path = '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_32.npy'
    # with open(path, 'rb') as f_user:
    #     user_review_emb = np.load(f_user)
    data = np.random.rand(100,2)
    num_clusters = 3
    overlap_users = [i for i in range(5)]
    centroids, cluster_labels = kmeans_temp(clusters=5,embeddings=data)
    print(cluster_labels)

    draw(embeddings=data,labels=cluster_labels,centers=centroids)
    # select_K(max_clusters=num_clusters,embeddings=user_review_emb)
    # overlap_users = [i for i in range(50)]
    # label_dict,cluster_overlap_smaples,cluster_labels = k_means(num_clusters=num_clusters,embeddings=data,overlap_users=overlap_users)
    # spectral_cluster(num_clusters,data,overlap_users)
    print('hh')
