""" Utilities
"""


import numpy as np
import scipy as sp
import os
import shutil
import tarfile
import scipy.stats as ss
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from itertools import combinations

from subfunc.showdata import *
from subfunc.munkres import Munkres


# =============================================================
# =============================================================
def w_to_directed(w, zero_tolerance=1e-10):
    """ Convert w to directed graph
    Args:
        w: [dim, dim, 2, group-pair]
        zero_tolerance: edges smaller than this value is undertermined
    Returns:
        wdir: directed w, NaN if not determined
    """

    num_dim, _, _, num_comb = w.shape
    wdir = w.copy()
    for c in range(num_comb):
        for i in range(num_dim):
            for j in range(num_dim):
                if np.abs(wdir[i, j, 0, c]) > np.abs(wdir[j, i, 1, c]):
                    wdir[j, i, 1, c] = 0
                elif np.abs(wdir[i, j, 0, c]) < np.abs(wdir[j, i, 1, c]):
                    wdir[i, j, 0, c] = 0
                elif (np.abs(wdir[i, j, 0, c]) == np.abs(wdir[j, i, 1, c])) and (np.abs(wdir[i, j, 0, c]) > zero_tolerance):
                    # cannot determine the direction
                    wdir[i, j, 0, c] = np.nan
                    wdir[j, i, 1, c] = np.nan

    return wdir


# =============================================================
# =============================================================
def w_permute(w, sort_idx, num_group, win=None):
    """ permute order of variables by sort_idx
    Args:
        w: [dim, dim, 2, group-pair]
        sort_idx: sorting index [dim, group]
        num_group: number of group
        win: [dim, dim, group]
    Returns:
        wdir: directed w, NaN if not determined
    """

    num_dim, _, _, num_combs = w.shape
    group_combs = list(combinations(np.arange(num_group), 2))

    w_perm = np.zeros_like(w)
    for mc in range(len(group_combs)):
        a = group_combs[mc][0]
        b = group_combs[mc][1]
        # a -> b
        w_mc = w[:, :, 0, mc].copy()
        w_mc = w_mc[sort_idx[:, a], :]
        w_mc = w_mc[:, sort_idx[:, b]]
        w_perm[:, :, 0, mc] = w_mc
        # b -> a
        w_mc = w[:, :, 1, mc].copy()
        w_mc = w_mc[sort_idx[:, b], :]
        w_mc = w_mc[:, sort_idx[:, a]]
        w_perm[:, :, 1, mc] = w_mc

    if win is not None:
        win_perm = np.zeros_like(win)
        for m in range(num_group):
            w_m = win[:, :, m].copy()
            w_m = w_m[sort_idx[:, m], :]
            w_m = w_m[:, sort_idx[:, m]]
            win_perm[:, :, m] = w_m
    else:
        win_perm = None

    return w_perm, win_perm


# =============================================================
# =============================================================
def w_threshold(w, thresh_ratio=0, comb_wise=False):
    """ Apply threshold to w
    Args:
        w: [dim, dim, 2, group-pair]
        thresh_ratio: Threshold ratio compared to the maximum absolute value
        comb_wise: Evaluate for each group-combination (True) or not (False)
    Returns:
        wthresh: thresholded w
    """

    num_node, _, _, num_comb = w.shape
    if comb_wise:
        wthresh = np.zeros_like(w)
        for c in range(num_comb):
            wc = w[:, :, :, c].copy()
            thval = np.max(np.abs(wc)) * thresh_ratio
            wc[np.abs(wc) <= thval] = 0
            wthresh[:, :, :, c] = wc
    else:
        wthresh = w.copy()
        thval = np.max(np.abs(wthresh)) * thresh_ratio
        wthresh[np.abs(wthresh) <= thval] = 0

    return wthresh


# =============================================================
# =============================================================
def eval_dag_bin(wtrue, west, flip=False):
    """ Evaluate estimated causal structure
    Args:
        wtrue: [dim, dim, 2, group-combinations] (binary)
        west: [ddim, dim, 2, group-combinations] (binary)
        flip: consider possible matrix-transpose of west
    Returns:
        f1:
        precision:
        recall:
        fpr:
        flipped: flipped the causal direction of west, or not
    """

    num_gcomb = wtrue.shape[3]

    wtrue_bin = wtrue.copy()
    wtrue_bin[(wtrue_bin != 0) & (~np.isnan(wtrue_bin))] = 1

    west_bin = west.copy()
    west_bin[np.isnan(west_bin)] = wtrue_bin[np.isnan(west_bin)]  # give true information for undetermined edges
    west_bin[(west_bin != 0) & (~np.isnan(west_bin))] = 1

    if flip:
        # flip for each group-pair if necessary
        flipped = np.zeros(num_gcomb, dtype=bool)
        for cn in range(num_gcomb):
            wtn = wtrue_bin[:, :, :, cn].copy()
            wen = west_bin[:, :, :, cn].copy()
            wen_flip = wen[:, :, ::-1]  # flip order
            wen_flip[:, :, 0] = wen_flip[:, :, 0].T  # transpose
            wen_flip[:, :, 1] = wen_flip[:, :, 1].T  # transpose
            f1 = f1_score(wtn.reshape(-1), wen.reshape(-1))
            f1_flip = f1_score(wtn.reshape(-1), wen_flip.reshape(-1))
            if f1_flip > f1:
                west_bin[:, :, :, cn] = west_bin[:, :, ::-1, cn]  # flip order
                west_bin[:, :, 0, cn] = west_bin[:, :, 0, cn].T  # transpose
                west_bin[:, :, 1, cn] = west_bin[:, :, 1, cn].T  # transpose
                flipped[cn] = True

    # evaluate
    wtrue_bin = wtrue_bin.reshape(-1)
    west_bin = west_bin.reshape(-1)

    precision = precision_score(wtrue_bin, west_bin)
    recall = recall_score(wtrue_bin, west_bin)
    f1 = f1_score(wtrue_bin, west_bin)
    tn, fp, fn, tp = confusion_matrix(wtrue_bin, west_bin).flatten()
    fpr = fp / (fp + tn)

    return f1, precision, recall, fpr, flipped


# =============================================================
# =============================================================
def eval_dag_bin_mat(wtrue, west, flip=False):
    """ Evaluate estimated causal structure
    Args:
        wtrue: [dim x group, dim x group] (binary)
        west: [dim x group, dim x group] (binary)
        flip: consider possible matrix-transpose of west
    Returns:
        f1:
        precision:
        recall:
        fpr:
        flipped: flipped the causal direction of west, or not
    """

    wtrue_bin = wtrue.copy()
    wtrue_bin[(wtrue_bin != 0) & (~np.isnan(wtrue_bin))] = 1

    west_bin = west.copy()
    west_bin[np.isnan(west_bin)] = wtrue_bin[np.isnan(west_bin)]  # give true information for undetermined edges
    west_bin[(west_bin != 0) & (~np.isnan(west_bin))] = 1

    # remove nan true edges
    wtrue_bin_nonnan = wtrue_bin[~np.isnan(wtrue_bin)].copy()
    west_bin_nonnan = west_bin[~np.isnan(wtrue_bin)].copy()

    # flip if necessary
    flipped = False
    if flip:
        west_bin_flip = west_bin.T
        west_bin_flip_nonnan = west_bin_flip[~np.isnan(wtrue_bin)].copy()
        f1 = f1_score(wtrue_bin_nonnan, west_bin_nonnan)
        f1_flip = f1_score(wtrue_bin_nonnan, west_bin_flip_nonnan)
        if f1_flip > f1:
            west_bin_nonnan = west_bin_flip_nonnan
            flipped = True

    # evaluate
    precision = precision_score(wtrue_bin_nonnan, west_bin_nonnan)
    recall = recall_score(wtrue_bin_nonnan, west_bin_nonnan)
    f1 = f1_score(wtrue_bin_nonnan, west_bin_nonnan)
    tn, fp, fn, tp = confusion_matrix(wtrue_bin_nonnan, west_bin_nonnan).flatten()
    fpr = fp / (fp + tn)

    return f1, precision, recall, fpr, flipped


# =============================================================
# =============================================================
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
         method: correlation method ('Pearson' or 'Spearman')
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
     """

    print('Calculating correlation...')

    x = x.copy().T
    y = y.copy().T
    dimx = x.shape[0]
    dimy = y.shape[0]

    # calculate correlation
    if method == 'Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dimy, dimy:]
    elif method == 'Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dimy, dimy:]
    else:
        raise ValueError
    if np.max(np.isnan(corr)):
        raise ValueError

    # sort
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dimy, dtype=int)
    for i in range(dimy):
        sort_idx[i] = indexes[i][1]
    sort_idx_other = np.setdiff1d(np.arange(0, dimx), sort_idx)
    sort_idx = np.concatenate([sort_idx, sort_idx_other])

    x_sort = x[sort_idx, :]

    # re-calculate correlation
    if method == 'Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dimy, dimy:]
    elif method == 'Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dimy, dimy:]
    else:
        raise ValueError

    return corr_sort, sort_idx, x_sort


# ===============================================================
# ===============================================================
def find_device():
    """find available device
    """
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        try:
            os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            print(memory_available)
            device = 'cuda:%d' % np.argmax(memory_available)
            print(device)
        except Exception as e:
            print(e)
            print('This is exception')
            device = torch.device('cuda')

    return device


# ===============================================================
# ===============================================================
def unzip(loadfile, unzipfolder, necessary_word='/storage'):
    """unzip trained model (loadfile) to unzipfolder
    """

    print('load: %s...' % loadfile)
    if loadfile.find(".tar.gz") > -1:
        if unzipfolder.find(necessary_word) > -1:
            if os.path.exists(unzipfolder):
                print('delete savefolder: %s...' % unzipfolder)
                shutil.rmtree(unzipfolder)  # remove folder
            archive = tarfile.open(loadfile)
            archive.extractall(unzipfolder)
            archive.close()
        else:
            assert False, "unzip folder doesn't include necessary word"
    else:
        if os.path.exists(unzipfolder):
            print('delete savefolder: %s...' % unzipfolder)
            shutil.rmtree(unzipfolder)  # remove folder
        os.makedirs(unzipfolder)
        src_files = os.listdir(loadfile)
        for fn in src_files:
            full_file_name = os.path.join(loadfile, fn)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, unzipfolder + '/')

    if not os.path.exists(unzipfolder):
        raise ValueError
