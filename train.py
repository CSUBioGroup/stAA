import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import torch
import time
import ot
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from model import VariationalEncoder, Regularizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score)
from sklearn.decomposition import PCA
import scanpy as sc


class stAA(nn.Module):
    def __init__(self, input_dim, n_clusters, hidden_dim=256, embed_dim=32, 
                 reg_hidden_dim_1=64, reg_hidden_dim_2=32,
                 clamp=0.01, epochs=1000) -> None:
        super(stAA, self).__init__()
        encoder = VariationalEncoder(in_channels=input_dim,
            hidden_channels=hidden_dim, out_channels=embed_dim)
        self.regularizer = Regularizer(embed_dim,
                                       reg_hidden_dim_2, reg_hidden_dim_1)
        self.graph = VGAE(encoder)
        self.ss_classifier = nn.Sequential(nn.Linear(embed_dim,
                                                     n_clusters, bias=False),
                                           nn.Sigmoid())
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.clamp = clamp

    def train_model(self, data,  ss_labels=None, position=None, data_save_path=None, 
                  method="mclust",labels=None, print_freq=100, 
                  lr=1e-5, W_a=0.4, W_x=0.6, refine=True, reso=0.5,
                  weight_decay=5e-05, reg_lr=1e-5, eval=True):
        loss_func = nn.CrossEntropyLoss()
        encoder_optimizer = torch.optim.Adam([{'params': self.graph.encoder.parameters()},
                                              {"params": self.ss_classifier.parameters()}
                                              ],
                                             lr=lr, weight_decay=weight_decay)
        regularizer_optimizer = torch.optim.Adam(self.regularizer.parameters(),
                                                 lr=reg_lr)
        data = data.cuda()
        if np.min(ss_labels) == 1:
            ss_labels = ss_labels - 1
        ss_labels = torch.tensor(ss_labels, dtype=torch.int64).cuda()
        for epoch in range(self.epochs):
            self.train()
            encoder_optimizer.zero_grad()

            z = self.graph.encode(data.x, data.train_pos_edge_index)

            for i in range(1):
                f_z = self.regularizer(z)
                r = torch.normal(
                    0.0, 1.0, [data.num_nodes, self.embed_dim]).cuda()
                f_r = self.regularizer(r)
                reg_loss = - f_r.mean() + f_z.mean()
                regularizer_optimizer.zero_grad()
                reg_loss.backward(retain_graph=True)
                regularizer_optimizer.step()

                for p in self.regularizer.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)
    
            f_z = self.regularizer(z)
            generator_loss = -f_z.mean()
            adj_recon_loss = self.graph.recon_loss(
                z, data.train_pos_edge_index) + (1 / data.num_nodes) * self.graph.kl_loss()
            adj_recon_loss = (adj_recon_loss + generator_loss) * W_a

            output_ss = self.ss_classifier(z)
            X_recon_loss = loss_func(output_ss, ss_labels) * W_x

            loss = X_recon_loss+adj_recon_loss
            loss.backward()
            encoder_optimizer.step()
            if (epoch+1) % print_freq == 0:
                print('Epoch: {:03d}, Loss: {:.4f}, ADJ Loss: {:.4f}, Gene Loss: {:.4f}'.format(
                    epoch+1, float(loss.item()), float(adj_recon_loss.item()), float(X_recon_loss.item())))

        completeness, hm, nmi, ari, z, pca_embedding, pred_label = self.eval_model(
                data, labels=labels, refine=refine, position=position,
                save_name=data_save_path, method=method, reso=reso)
        res = {}
        res["embedding"] = z
        res["pred_label"] = pred_label
        res["embedding_pca"] = pca_embedding
        if eval == True:
            res["nmi"] = nmi
            res["ari"] = ari
            res["completeness"] = completeness
            res["hm"] = hm

        return res


    @torch.no_grad()
    def eval_model(self, data, labels=None, refine=False, reso=0.5,
                   position=None, save_name=None, method="mclust"):
        self.eval()
        z = self.graph.encode(data.x, data.train_pos_edge_index)
        pca_input = dopca(z.cpu().numpy(), dim=20)
        if method=="mclust":
            pred_mclust = mclust_R(embedding=pca_input,
                                num_cluster=self.n_clusters)
        if method == "louvain":
            adata_tmp=sc.AnnData(pca_input)
            sc.pp.neighbors(adata_tmp, n_neighbors=20)
            sc.tl.louvain(adata_tmp, resolution=reso, random_state=0)
            pred_mclust=adata_tmp.obs['louvain'].astype(int).to_numpy()
        if refine:
            pred_mclust = refine_label(pred_mclust, position, radius=50)

        if labels is not None:
            label_df = pd.DataFrame({"True": labels,
                                    "Pred": pred_mclust}).dropna()
            # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
            completeness = completeness_score(
                label_df["True"], label_df["Pred"])
            hm = homogeneity_score(label_df["True"], label_df["Pred"])
            nmi = v_measure_score(label_df["True"], label_df["Pred"])
            ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        else:
            completeness, hm, nmi, ari = 0, 0, 0, 0

        if (save_name is not None):
            np.save(save_name+"pca.npy", pca_input)
            np.save(save_name+"embedding.npy", z.cpu().numpy())
            if (labels is not None):
                pd.DataFrame({"True": labels, 
                            "Pred": pred_mclust}).to_csv(save_name+'types.txt')
            else:
                pd.DataFrame({
                            "Pred": pred_mclust}).to_csv(save_name+'types.txt')

        return completeness, hm, nmi, ari, z.cpu().numpy(), pca_input, pred_mclust


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=42)
    X_10 = pcaten.fit_transform(X)
    return X_10


def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(
        embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')
    return mclust_res



def refine_label(label, position, 
                 radius=50):
    new_type = []

    # calculate distance
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, radius+1):
            neigh_type.append(label[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type
