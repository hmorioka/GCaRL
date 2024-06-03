""" Evaluation
    Main script for evaluating the model trained by gcarl_training.py
"""


import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib.colors import ListedColormap
import colorcet as cc
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations
import faiss
import matplotlib.pyplot as plt

from subfunc.generate_dataset import generate_dataset, wg_to_w
from subfunc.threedident_dataset import RawThreeDIdentDataset
from gcarl import gcarl, utils
from gcarl_3dident import gcarl as gcarl_3dident

from subfunc.showdata import *

# parameters ==================================================
# =============================================================

eval_dir_base = './storage'

eval_dir = os.path.join(eval_dir_base, 'model')

parmpath = os.path.join(eval_dir, 'parm.pkl')
savefile = eval_dir.replace('.tar.gz', '') + '.pkl'

thresh_ratio = np.arange(0, 1.05, 0.05)
load_ema = True  # recommended unless the number of iterations was not enough
device = 'cpu'


# =============================================================
# =============================================================
if eval_dir.find('.tar.gz') >= 0:
    unzipfolder = './storage/temp_unzip'
    utils.unzip(eval_dir, unzipfolder)
    eval_dir = unzipfolder
    parmpath = os.path.join(unzipfolder, 'parm.pkl')

modelpath = os.path.join(eval_dir, 'model.pt')

# load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_group = model_parm['num_group']
num_dim = model_parm['num_dim']
num_data = model_parm['num_data']
lam1 = model_parm['lam1']
lam2 = model_parm['lam2']
lamin1 = model_parm['lamin1']
lamin2 = model_parm['lamin2']
ar_alpha = model_parm['ar_alpha']
ar_beta = model_parm['ar_beta']
num_neighbor = model_parm['num_neighbor']
num_neighbor_in = model_parm['num_neighbor_in']
dag = model_parm['dag'] if 'dag' in model_parm else False
dist_type = model_parm['dist_type'] if 'dist_type' in model_parm else 'Laplace'
num_layer = model_parm['num_layer']
num_h_nodes = model_parm['num_h_nodes']
num_hz_nodes = model_parm['num_hz_nodes']
num_hp_nodes = model_parm['num_hp_nodes']
phi_type = model_parm['phi_type']
phi_share = model_parm['phi_share']
num_latent = model_parm['num_latent'] if 'num_latent' in model_parm else None
model_type = model_parm['model_type'] if 'model_type' in model_parm else 'crl'
beta = model_parm['beta'] if 'beta' in model_parm else None
apply_pca = model_parm['apply_pca'] if 'apply_pca' in model_parm else False
data_path = model_parm['data_path'] if 'data_path' in model_parm else None
random_seed = model_parm['random_seed']


# generate sensor signal --------------------------------------
x, s, A1, A2, Ain1, Ain2 = generate_dataset(num_group=num_group,
                                            num_dim=num_dim,
                                            num_data=num_data,
                                            num_layer=num_layer,
                                            lam1_range=lam1,
                                            lam2_range=lam2,
                                            lamin1_range=lamin1,
                                            lamin2_range=lamin2,
                                            ar_alpha=ar_alpha,
                                            ar_beta=ar_beta,
                                            num_neighbor=num_neighbor,
                                            num_neighbor_in=num_neighbor_in,
                                            dag=dag,
                                            dist_type=dist_type,
                                            num_latent=num_latent,
                                            random_seed=random_seed)

# preprocessing
if apply_pca and (len(num_h_nodes) > 0):
    for m in range(num_group):
        pca = PCA(whiten=True)
        x[:, m, :] = pca.fit_transform(x[:, m, :])
elif apply_pca and (len(num_h_nodes) == 0):
    x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

if data_path is not None:
    # load latents of images
    z = np.load(os.path.join(data_path, "raw_latents.npy"))
    s_range = np.percentile(s.reshape(-1), [0.01, 99.99])
    s_norm = 2 * ((s - s_range[0]) / (s_range[1] - s_range[0]) - 0.5)
    index_fn = faiss.IndexFlatL2(z.shape[1])
    index_fn.add(z)
    distance_s_to_z, index_s_to_z = index_fn.search(s_norm.reshape([-1, s_norm.shape[-1]]), 1)
    index_s_to_z = index_s_to_z.reshape([s_norm.shape[0], s_norm.shape[1]])

    # load whole images
    dataset = RawThreeDIdentDataset(root=data_path,
                                    transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.4363, 0.2818, 0.3045], std=[0.1197, 0.0734, 0.0919]  # Yao2023
                                                # mean=[0.3292, 0.3278, 0.3215], std=[0.0778, 0.0776, 0.0771]  # CL-ICA
                                            ),
                                        ]))
    data_loader = DataLoader(dataset,
                             batch_size=int(2**10),
                             shuffle=False,
                             drop_last=False,
                             num_workers=0)
    x_img = torch.zeros([len(z), 3, 224, 224], device='cpu')
    for step, batch in enumerate(data_loader):
        sys.stdout.write('\rLoading x... %d / %d' % (step * data_loader.batch_size, len(z)))
        sys.stdout.flush()
        idx = batch[1].detach().cpu().numpy()
        x_img[idx, :] = batch[0]
    sys.stdout.write('\r\n')


# build model ------------------------------------------------
# ------------------------------------------------------------

if data_path is None:
    # define network
    model = gcarl.Net(num_group=num_group,
                      num_xdim=[x.shape[2]] * num_group,
                      h_sizes=num_h_nodes if num_h_nodes is not None else [num_dim],
                      hz_sizes=num_hz_nodes if num_hz_nodes is not None else [num_dim],
                      hp_sizes=num_hp_nodes,
                      phi_type=phi_type,
                      phi_share=phi_share)
    model = model.to(device)
    model.eval()

    # load parameters
    print('Load trainable parameters from %s...' % modelpath)
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if load_ema:
        model.load_state_dict(checkpoint['ema_state_dict'])

    # feedforward for h
    x_torch = torch.from_numpy(x.astype(np.float32)).to(device)
    logits, h, phi = model(x_torch)

    # feedforward for h_ast
    x_ast = np.zeros_like(x)
    for m in range(num_group):
        x_ast[:, m, :] = x[np.random.permutation(x.shape[0]), m, :]
    x_torch = torch.from_numpy(x_ast.astype(np.float32)).to(device)
    logits_ast, h_ast, phi_ast = model(x_torch)

    # convert to numpy
    h_val = h.detach().cpu().numpy()
    phi_val = phi.detach().cpu().numpy()
    pred_val = (torch.cat([logits, logits_ast]) > 0).cpu().numpy().astype(float)
    label_val = np.concatenate([np.ones(x.shape[0]), np.zeros(x.shape[0])])

else:  # 3dIdent high-dimensional image
    # define network
    model = gcarl_3dident.Net(num_group=num_group,
                              num_hdim=num_h_nodes[-1],
                              hz_sizes=num_hz_nodes if num_hz_nodes is not None else [num_dim],
                              hp_sizes=num_hp_nodes,
                              phi_type=phi_type)
    device = utils.find_device()  # find GPU
    model = model.to(device)
    model.eval()

    # load parameters
    print('Load trainable parameters from %s...' % modelpath)
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if load_ema:
        model.load_state_dict(checkpoint['ema_state_dict'])

    # feedforward
    logits = np.zeros(num_data)
    logits_ast = np.zeros(num_data)
    h = np.zeros([num_data, num_group, num_h_nodes[-1]], np.float32)
    h_ast = np.zeros([num_data, num_group, num_h_nodes[-1]], np.float32)
    batch_size = int(2**9)
    with torch.no_grad():
        for i in range(int(np.ceil(num_data / batch_size))):
            # torch.cuda.empty_cache()
            sys.stdout.write('\rFeedforwading h... %d / %d' % (batch_size * i, num_data))
            sys.stdout.flush()
            idx = np.arange(batch_size * i, min(batch_size * (i + 1), num_data))
            x0 = torch.stack([x_img[index_s_to_z[idx, m], :] for m in range(num_group)], dim=1)
            xast = x0.clone()
            fix_group = np.random.randint(num_group)  # single group has the same value to x0
            for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
                idx_ast = np.random.choice(num_data, batch_size)
                xast[:, m, :] = x_img[index_s_to_z[idx_ast, m], :]
            logits_i, h_i, _ = model(torch.cat([x0, xast], dim=0).to(device))
            logits_i = logits_i.detach().cpu().numpy()
            h_i = h_i.detach().cpu().numpy()
            logits[idx] = logits_i[:batch_size]
            h[idx, :, :] = h_i[:batch_size, :]
            logits_ast[idx] = logits_i[batch_size:]
            h_ast[idx, :, :] = h_i[batch_size:, :]
    sys.stdout.write('\r\n')

    h_val = h
    pred_val = np.concatenate([logits, logits_ast]) > 0
    label_val = np.concatenate([np.ones(x.shape[0]), np.zeros(x.shape[0])])

# causal graph
if phi_type.startswith('lap'):
    west = - model.w.to('cpu').detach().numpy()
    west2 = model.w2.to('cpu').detach().numpy()
    west[:, :, 0, :] = west[:, :, 0, :] * np.abs(west2)
    west[:, :, 1, :] = west[:, :, 1, :] * np.abs(west2).transpose((1, 0, 2))
else:
    west = - model.w.to('cpu').detach().numpy()

# evaluate outputs --------------------------------------------
# -------------------------------------------------------------

# classification accuracy
accu = accuracy_score(pred_val, label_val)

# correlation for each group
corrmat = np.zeros([s.shape[2], s.shape[2], num_group])
sort_idx = np.zeros([s.shape[2], num_group], dtype=int)
correlation_measure = 'Pearson' if data_path is None else 'Spearman'
for m in range(num_group):
    corrmat[:, :, m], sort_idx[:, m], _ = utils.correlation(h_val[:, m, :], s[:, m, :], correlation_measure)
meanabscorr = np.mean(np.abs(np.diagonal(corrmat)), axis=1)

# causal structure
if phi_type.startswith('lap'):
    wtrue = A1
    wintrue = Ain1
else:
    wtrue = A1 * A2
    wintrue = Ain1 * Ain2

# permutation
west_perm, _ = utils.w_permute(west, sort_idx=sort_idx, num_group=num_group)
# determine direction
west_dir = utils.w_to_directed(west_perm)

wtrue_eval = wtrue.copy()
west_eval = west_dir.copy()
f1 = np.zeros(len(thresh_ratio))
precision = np.zeros(len(thresh_ratio))
recall = np.zeros(len(thresh_ratio))
fpr = np.zeros(len(thresh_ratio))
# flipped = np.zeros([len(thresh_ratio), wtrue_eval.shape[-1]])
flipped = np.zeros([len(thresh_ratio)])
do_flip = False if data_path is None else True
for i in range(len(thresh_ratio)):
    west_thresh = utils.w_threshold(west_eval, thresh_ratio=thresh_ratio[i], comb_wise=True)  # thresholding
    f1[i], precision[i], recall[i], fpr[i], flipped[i] = utils.eval_dag_bin_mat(wg_to_w(wtrue_eval, num_group=num_group),
                                                                                wg_to_w(west_thresh, num_group=num_group),
                                                                                flip=do_flip)  # whole-graph transpose

# display results
print('Result...')
print('    accuracy  : %7.4f [percent]' % (accu * 100))
print(' correlation  : %7.4f' % np.mean(meanabscorr))
print('          F1  : (max) %7.4f (th=%g)' % (np.max(f1), thresh_ratio[np.argmax(f1)]))
print('   precision  : (max) %7.4f (th=%g)' % (np.max(precision), thresh_ratio[np.argmax(precision)]))
print('      recall  : (max) %7.4f (th=%g)' % (np.max(recall), thresh_ratio[np.argmax(recall)]))

# save results
result = {'accu': accu if 'accu' in locals() else None,
          'corrmat': corrmat if 'corrmat' in locals() else None,
          'meanabscorr': meanabscorr,
          'sort_idx': sort_idx,
          'f1': f1,
          'precision': precision,
          'recall': recall,
          'fpr': fpr,
          'flipped': flipped,
          'num_group': num_group,
          'num_dim': x.shape[2],
          'thresh_ratio': thresh_ratio,
          'modelpath': modelpath}

print('Save results...')
with open(savefile, 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


# visualization: correlation s vs h
for m in range(num_group):
    showmat(corrmat[:, :, m],
            yticklabel=np.arange(0, corrmat.shape[0]),
            xticklabel=sort_idx[:, m].astype(np.int32),
            ylabel='True',
            xlabel='Estimated')


# visualize adjacency matrix in a single plot
group_combs = list(combinations(np.arange(num_group), 2))
plt.figure(figsize=(8 * 1.2, 6 * 1.5))
wdisp = wtrue.copy()
for m in range(num_group):
    plt.subplot(num_group, num_group, num_group * m + m + 1)
    plt.imshow(wintrue[:, :, m], interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
    plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
    plt.axis('off')
for c in range(wtrue.shape[3]):
    for d in range(2):
        m, mk = group_combs[c]
        if d == 0:
            plt.subplot(num_group, num_group, num_group * m + mk + 1)
        else:
            plt.subplot(num_group, num_group, num_group * mk + m + 1)
        plt.imshow(wdisp[:, :, d, c], interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
        plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
        plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(8 * 1.2, 6 * 1.5))
wdisp = west_perm.copy()
# wdisp = west_dir.copy()
# wdisp = - west_thresh.copy()
for c in range(wtrue.shape[3]):
    for d in range(2):
        m, mk = group_combs[c]
        if d == 0:
            plt.subplot(num_group, num_group, num_group * m + mk + 1)
        else:
            plt.subplot(num_group, num_group, num_group * mk + m + 1)
        plt.imshow(wdisp[:, :, d, c], interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
        plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
        plt.axis('off')
plt.tight_layout()

