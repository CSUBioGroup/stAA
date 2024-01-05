import squidpy as sq
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score,  silhouette_score, homogeneity_completeness_v_measure, davies_bouldin_score)
from sklearn.metrics.cluster import adjusted_rand_score
import random
import os
import numpy as np
import torch
from train import stAA
from data import Adata2Torch_data, Spatial_Dis_Cal, process_adata, read_data, get_initial_label
import warnings
warnings.filterwarnings("ignore")


seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def eval_embedding(pred, embedding=None):
    sc = silhouette_score(embedding, pred, metric='euclidean')
    db = davies_bouldin_score(embedding, pred)
    return sc, db


def run_stAA(adata, n_clusters, graph_mode="knn", cluster_method="mclust",
             refine=True, data_save_path="./", true_labels=None, eval=True):
    # if cluster_method="louvain", "n_clusters" represents the resolution
    adata = process_adata(adata)

    if graph_mode in ["knn", "KNN"]:
        Spatial_Dis_Cal(adata, knn_dis=5, model="KNN")
    else:
        Spatial_Dis_Cal(adata, rad_dis=graph_mode)

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError(
            "Please Compute Spatial Network using Spatial_Dis_Cal function first!")
    # Process the data
    data = Adata2Torch_data(adata)
    ss_labels = get_initial_label(adata, method=cluster_method,
                                  n_clusters=n_clusters)
    reso = n_clusters
    if cluster_method == "mclust":
        n_clusters = n_clusters
    else:
        n_clusters = len(set(ss_labels))
    model = stAA(input_dim=data.x.shape[1], epochs=1000,
                 hidden_dim=256, embed_dim=128, n_clusters=n_clusters).cuda()
    res = model.train_model(
        data, method=cluster_method, refine=refine,
        position=adata.obsm['spatial'], eval=eval, reso=reso,
        ss_labels=ss_labels, data_save_path=data_save_path,
        labels=true_labels)
    return res


adata = sq.datasets.slideseqv2()
adata.var_names_make_unique()
# train model and cluster
res = run_stAA(adata, n_clusters=10,
               cluster_method="mclust", refine=False,
               graph_mode=40, eval=False,
               data_save_path="./")
print(res["embedding"])
print(res["pred_label"])
