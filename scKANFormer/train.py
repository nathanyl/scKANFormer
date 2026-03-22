import random
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform

from scKANFormer_model import scTrans_model as create_model
from multi_imbalance.resampling.mdo import MDO
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)


# def add_zinb_noise(data, noise_fraction=0.1):
#     """根据平均值动态地添加零膨胀负二项分布噪声并显示进度条"""
#     n_samples, n_features = data.shape
#     noisy_data = data.copy()
#     num_noisy_samples = int(n_samples * noise_fraction)
#     noisy_indices = np.random.choice(n_samples, num_noisy_samples, replace=False)
#
#     for i in tqdm(noisy_indices, desc="Adding Dynamic ZINB Noise"):
#         for j in range(n_features):
#             mean = data[i, j]
#             if mean > 0:
#                 # 动态调整零膨胀概率和负二项分布参数
#                 pi = min(0.1, 1 / (mean + 1))  # 根据平均值动态调整零膨胀概率
#                 #theta = 0.8  # 根据平均值动态调整负二项分布参数
#                 theta = max(0.1, mean / 10)
#                 # 判断是否为零膨胀
#                 if np.random.rand() < pi:
#                     noisy_data[i, j] = 0
#                 else:
#                     p = mean / (mean + theta)
#                     noisy_data[i, j] = np.random.negative_binomial(theta, p)
#             else:
#                 noisy_data[i, j] = 0
#     return noisy_data
# def add_negbin_noise(data, noise_fraction=0.15):
#     """根据平均值动态地添加负二项分布噪声并显示进度条"""
#     n_samples, n_features = data.shape
#     noisy_data = data.copy()
#     num_noisy_samples = int(n_samples * noise_fraction)
#     noisy_indices = np.random.choice(n_samples, num_noisy_samples, replace=False)
#
#     for i in tqdm(noisy_indices, desc="Adding Dynamic Negative Binomial Noise"):
#         for j in range(n_features):
#             mean = data[i, j]
#             if mean > 0:
#                 # 动态调整负二项分布参数
#                 theta = max(0.1, mean / 10)
#                 p = mean / (mean + theta)
#                 noisy_data[i, j] = np.random.negative_binomial(theta, p)
#             else:
#                 noisy_data[i, j] = 0
#     return noisy_data
###
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder

# def add_gaussian_noise(data, noise_level):
#     """
#     给数据添加高斯噪声。
#
#     参数:
#     data (np.array): 输入数据。
#     noise_level (float): 高斯噪声的标准差。
#
#     返回:
#     np.array: 添加高斯噪声后的数据。
#     """
#     noise = np.random.normal(0, noise_level, data.shape)
#     return data + noise

# def splitDataSetG(adata, label_name='Celltype', tr_ratio=0.7, noise_level=0.15, noise_proportion=0.1):
#     """
#     将数据集划分为训练集和测试集，并按比例添加高斯噪声。
#
#     参数:
#     adata: 注释数据矩阵。
#     label_name (str): 标签列的名称。
#     tr_ratio (float): 训练数据的比例。
#     noise_level (float): 高斯噪声的标准差。
#     noise_proportion (float): 添加噪声的数据比例。
#
#     返回:
#     tuple: 训练集和验证集，以及标签信息和基因名称。
#     """
#     label_encoder = LabelEncoder()
#     el_data = adata.to_df()
#     el_data[label_name] = adata.obs[label_name].astype('str')
#     genes = el_data.columns.values[:-1]
#     el_data = np.array(el_data)
#     el_data[:, -1] = label_encoder.fit_transform(el_data[:, -1])
#     inverse = label_encoder.inverse_transform(range(0, np.max(el_data[:, -1]) + 1))
#     el_data = el_data.astype(np.float32)
#     el_data = balance_populations(data=el_data)
#
#     # 按比例添加高斯噪声
#     if noise_level > 0 and noise_proportion > 0:
#         num_samples = int(el_data.shape[0] * noise_proportion)
#         indices = np.random.choice(el_data.shape[0], num_samples, replace=False)
#         el_data[indices, :-1] = add_gaussian_noise(el_data[indices, :-1], noise_level)
#
#     n_genes = len(el_data[0]) - 1
#     train_size = int(len(el_data) * tr_ratio)
#     train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size, len(el_data) - train_size])
#     exp_train = torch.from_numpy(np.array(train_dataset)[:, :n_genes].astype(np.float32))
#     label_train = torch.from_numpy(np.array(train_dataset)[:, -1].astype(np.int64))
#     exp_valid = torch.from_numpy(np.array(valid_dataset)[:, :n_genes].astype(np.float32))
#     label_valid = torch.from_numpy(np.array(valid_dataset)[:, -1].astype(np.int64))
#     return exp_train, label_train, exp_valid, label_valid, inverse, genes
###
def splitDataSet(adata,label_name='Celltype', tr_ratio= 0.7): 
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data = el_data)
    # 尝试添加噪声
    #el_data[:, :-1] = add_negbin_noise(el_data[:, :-1])

    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(np.array(valid_dataset)[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(np.array(valid_dataset)[:,-1].astype(np.int64))
    return exp_train, label_train, exp_valid, label_valid, inverse, genes


# def splitDataSetNew(adata,label_name='Celltype', tr_ratio= 0.7):
#     """
#     Split data set into training set and test set.
#
#     """
#     label_encoder = LabelEncoder()
#     el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
#     el_data[label_name] = adata.obs[label_name].astype('str')
#     #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
#     genes = el_data.columns.values[:-1]
#     el_data = np.array(el_data)
#     # el_data = np.delete(el_data,-1,axis=1)
#     el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
#     inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
#     el_data = el_data.astype(np.float32)
#     #新的采样方式
#     mdo = MDO(k=10, k1_frac=0.18, seed=0, prop=0.99)
#     el_data[:, :-1] = add_zinb_noise(el_data[:, :-1])
#     el_data = balance_populations(data = el_data)
#     n_genes = len(el_data[1])-1
#     train_size = int(len(el_data) * tr_ratio)
#     train_indices = np.random.choice(len(el_data), size=train_size, replace=False)
#     valid_indices = list(set(range(len(el_data))) - set(train_indices))
#
#     train_dataset = el_data[train_indices]
#     valid_dataset = el_data[valid_indices]
#
#     X_train = train_dataset[:, :n_genes]
#     y_train = train_dataset[:, -1]
#
#     X_train_resampled, y_train_resampled = mdo.fit_resample(X_train, y_train)
#     #train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
#     exp_train = torch.from_numpy(X_train_resampled.astype(np.float32))
#     label_train = torch.from_numpy(y_train_resampled.astype(np.int64))
#     exp_valid = torch.from_numpy(valid_dataset[:, :n_genes].astype(np.float32))
#     label_valid = torch.from_numpy(valid_dataset[:, -1].astype(np.int64))
#     return exp_train, label_train, exp_valid, label_valid, inverse, genes
###################################################################################
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split, TensorDataset
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# import numpy as np
# from scipy.sparse import csr_matrix

# class ComplexVAE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(ComplexVAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )
#         self.fc_mu = nn.Linear(256, latent_dim)
#         self.fc_logvar = nn.Linear(256, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, input_dim),
#             nn.Sigmoid(),
#         )
#
#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_logvar(h)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         return self.decoder(z)
#
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar
#
#     def loss_function(self, recon_x, x, mu, logvar):
#         BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return BCE + KLD
#
# def train_vae(vae, train_dataloader, epochs, learning_rate=1e-3):
#     optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
#     vae.train()
#     for epoch in range(epochs):
#         train_loss = 0
#         for batch in train_dataloader:
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = vae(batch[0])
#             loss = vae.loss_function(recon_batch, batch[0], mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#         print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_dataloader.dataset)}")

def split(adata, label_name='Celltype', tr_ratio=0.7, latent_dim=10, epochs=20, batch_size=64):
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(adata.X.toarray(), index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    el_data[:, -1] = label_encoder.fit_transform(el_data[:, -1])
    inverse = label_encoder.inverse_transform(range(0, np.max(el_data[:, -1]) + 1))
    el_data = el_data.astype(np.float32)

    n_genes = len(el_data[1]) - 1

    dataset = TensorDataset(torch.tensor(el_data[:, :-1], dtype=torch.float32), torch.tensor(el_data[:, -1], dtype=torch.long))
    train_size = int(len(dataset) * tr_ratio)
    train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    vae = ComplexVAE(input_dim=n_genes, latent_dim=latent_dim)
    train_vae(vae, train_dataloader, epochs)

    # Generate new samples using VAE
    vae.eval()
    with torch.no_grad():
        z = torch.randn(len(dataset), latent_dim)
        generated_samples = vae.decode(z).numpy()

    # Add generated samples to the original dataset
    augmented_data = np.concatenate((el_data[:, :-1], generated_samples), axis=0)
    augmented_labels = np.concatenate((el_data[:, -1], el_data[:, -1]), axis=0)  # duplicate labels

    # Split data into training and validation sets
    augmented_data = np.column_stack((augmented_data, augmented_labels))
    train_size = int(len(augmented_data) * tr_ratio)
    train_dataset, valid_dataset = random_split(augmented_data, [train_size, len(augmented_data) - train_size])

    exp_train = torch.tensor(np.array(train_dataset)[:, :n_genes], dtype=torch.float32)
    label_train = torch.tensor(np.array(train_dataset)[:, -1], dtype=torch.int64)
    exp_valid = torch.tensor(np.array(valid_dataset)[:, :n_genes], dtype=torch.float32)
    label_valid = torch.tensor(np.array(valid_dataset)[:, -1], dtype=torch.int64)

    return exp_train, label_train, exp_valid, label_valid, inverse, genes
#################################################################################
def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    #print(model)
    loss_function = torch.nn.CrossEntropyLoss() 
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    #focal_loss = FocalLoss(alpha = 0.3, gamma=2, reduction='mean')
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        # 损失函数改动
        #F_loss = focal_loss(pred, label.to(device))
        #loss = #F_loss #+ loss
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()

def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    #focal_loss = FocalLoss(alpha=0.3, gamma=2, reduction='mean')
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        # 损失函数改动
        #F_loss = focal_loss(pred, labels.to(device))
        #loss = F_loss #+ loss

        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=16, embed_dim=24,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    #train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0) > 4]
        mask = mask[:,np.sum(mask,axis=0) > 4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        #print(mask.shape)
        print('Mask loaded!')
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train)+1)
    #print(num_classes)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True, drop_last=True)
    model = create_model(num_classes=num_classes, num_genes=len(exp_train[0]),  mask=mask, embed_dim=embed_dim, depth=depth, num_heads=num_heads, has_logits=False).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name) 
    print('Model builded!')
    print(model)
    pg = [p for p in model.parameters() if p.requires_grad]  
    #optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.NAdam(pg, lr=lr,  weight_decay=5E-5, momentum_decay = 0.9)
    #optimizer = optim.AdamW(pg, lr=lr,  weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step() 
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if platform.system().lower() == 'windows':
            torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
        else:
            torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
    print('Training finished!')

#train(adata, gmt_path, pre_weights, batch_size=8, epochs=20)


