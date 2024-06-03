"""Data generation"""

import sys
import os
import numpy as np
from scipy.special import comb
from scipy.stats import norm, truncexpon
import networkx as nx
from itertools import combinations
from subfunc.showdata import *
# from subfunc.sergio.sergio import sergio


# =============================================================
# =============================================================
def generate_dataset(num_group,
                     num_dim,
                     num_data,
                     num_layer,
                     lam1_range=None,
                     lam2_range=None,
                     lamin1_range=None,
                     lamin2_range=None,
                     ar_alpha=None,
                     ar_beta=None,
                     num_neighbor=None,
                     num_neighbor_in=None,
                     num_latent=None,
                     dag=False,
                     dist_type='Gauss',
                     negative_slope=0.2,
                     x_limit=1e2,
                     random_seed=0):
    """Generate artificial data.
    Args:
        num_group: number of groups
        num_dim: number of variables in a single group
        num_data: number of data
        num_layer: number of layers of mixing-MLP
        lam1_range: range of lam1
        lam2_range: range of lam2
        lamin1_range: range of lamin1
        lamin2_range: range of lamin2
        ar_alpha: AR parameter
        ar_beta: AR parameter
        num_neighbor: number of neighbors of a variable in a single group-pair
        num_neighbor_in: number of neighbors of a variable in the same group
        num_latent: number of latent confounders (included in num_dim)
        dag: DAG(True) or cyclic(False)
        dist_type: noise distribution type
        negative_slope: negative slope of leaky ReLU
        x_limit: if x exceed this range, re-generate it
        random_seed: random seed
    Returns:
        x: observed signals [data, group, dim]
        s: latent variables  [data, group, dim]
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    """

    stable_flag = False
    cnt = 0
    while not stable_flag:
        # change random seed
        random_seed = random_seed + num_data + num_layer*100 + cnt*10000

        if dist_type == 'GRN':
            # generate graph
            lam1, lam2, lamin1, lamin2 = gen_net_sergio(num_group,
                                                        num_dim,
                                                        lam1_range=lam1_range,
                                                        lam2_range=lam2_range,
                                                        lamin1_range=lamin1_range,
                                                        lamin2_range=lamin2_range,
                                                        num_neighbor=num_neighbor,
                                                        num_neighbor_in=num_neighbor_in,
                                                        num_latent=num_latent,
                                                        random_seed=random_seed)

            # generate s
            lam1mat = wg_to_w(wg=lam1, num_group=num_group, wgin=lamin1)
            G = nx.from_numpy_array(lam1mat, create_using=nx.DiGraph)
            s = simulate_sergio(G, num_data, hill=6, mr_range=(0.25, 0.75), seed=random_seed).T
            # NOTE: need to install and import SERGIO by yourself
            s = s.reshape([num_data, num_group, num_dim])

        else:
            # generate graph
            if dag:
                lam1, lam2, lamin1, lamin2 = gen_net_dag(num_group=num_group,
                                                         num_dim=num_dim,
                                                         lam1_range=lam1_range,
                                                         lam2_range=lam2_range,
                                                         lamin1_range=lamin1_range,
                                                         lamin2_range=lamin2_range,
                                                         num_neighbor=num_neighbor,
                                                         num_neighbor_in=num_neighbor_in,
                                                         random_seed=random_seed)
            else:
                lam1, lam2, lamin1, lamin2 = gen_net(num_group=num_group,
                                                     num_dim=num_dim,
                                                     lam1_range=lam1_range,
                                                     lam2_range=lam2_range,
                                                     lamin1_range=lamin1_range,
                                                     lamin2_range=lamin2_range,
                                                     num_neighbor=num_neighbor,
                                                     num_neighbor_in=num_neighbor_in,
                                                     num_latent=num_latent,
                                                     random_seed=random_seed)

            # generate s
            if dist_type == 'Gauss':
                s = gen_s(lam1=lam1,
                          lam2=lam2,
                          lamin1=lamin1,
                          lamin2=lamin2,
                          lam1_mean=np.mean(lam1_range),
                          num_data=num_data,
                          ar_alpha=ar_alpha,
                          dag=dag,
                          random_seed=random_seed)
            elif dist_type == 'Laplace':
                s = gen_s_lap(lam1=lam1,
                              lamin1=lamin1,
                              lam1_mean=np.mean(lam1_range),
                              num_data=num_data,
                              ar_alpha=ar_alpha,
                              ar_beta=ar_beta,
                              dag=dag,
                              random_seed=random_seed)

        # normalize
        s_norm = (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)

        # mask latent confounders
        if num_latent is not None:
            num_observe = num_dim - num_latent
            assert num_dim % num_observe == 0
            pick_interval = int(num_dim / num_observe)
            s = s[:, :, pick_interval - 1::pick_interval]
            s_norm = s_norm[:, :, pick_interval - 1::pick_interval]
            lam1 = lam1[pick_interval - 1::pick_interval, :, :, :][:, pick_interval - 1::pick_interval, :, :]
            lam2 = lam2[pick_interval - 1::pick_interval, :, :, :][:, pick_interval - 1::pick_interval, :, :]
            lamin1 = lamin1[pick_interval - 1::pick_interval, :, :][:, pick_interval - 1::pick_interval, :]
            lamin2 = lamin2[pick_interval - 1::pick_interval, :, :][:, pick_interval - 1::pick_interval, :]

        if num_layer > 0:
            # generate MLP
            mlplayer = [gen_mlp(num_dim=s_norm.shape[2],
                                num_layer=num_layer,
                                random_seed=random_seed + k) for k in range(num_group)]

            # generate x
            x = gen_x(s=s_norm,
                      mlplayer=mlplayer,
                      negative_slope=negative_slope)
        else:
            x = s.copy()

        # check stability
        x_max = np.max(np.abs(x))
        if x_max < x_limit:
            stable_flag = True

        cnt = cnt + 1

    return x, s, lam1, lam2, lamin1, lamin2


# =============================================================
# =============================================================
def gen_net(num_group,
            num_dim,
            lam1_range=None,
            lam2_range=None,
            lamin1_range=None,
            lamin2_range=None,
            num_neighbor=2,
            num_neighbor_in=1,
            num_latent=None,
            norm_by_num_parents=True,
            random_seed=0):
    """Generate graph.
    Args:
        num_group: number of groups
        num_dim: number of variables in a single group
        lam1_range: range of inter-lambda1
        lam2_range: range of inter-lambda2
        lamin1_range: range of intra-lambda1
        lamin2_range: range of intra-lambda2
        num_neighbor: number of neighbors of a variable in a single group-pair
        num_neighbor_in: number of neighbors of a variable in the same group
        num_latent: number of latent confounders (included in num_dim)
        norm_by_num_parents: normalize by number of parents, or not
        random_seed: random seed
    Returns:
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    """

    print("Generating graph (cyclic)...")

    if lam1_range is None:
        lam1_range = [1, 1]
    if lam2_range is None:
        lam2_range = [1, 1]
    if lamin1_range is None:
        lamin1_range = [1, 1]
    if lamin2_range is None:
        lamin2_range = [1, 1]

    # initialize random generator
    np.random.seed(random_seed)

    group_combs = list(combinations(np.arange(num_group), 2))
    num_comb = int(comb(num_group, 2))

    # inter-group
    lam1 = np.zeros([num_dim, num_dim, 2, num_comb])
    for cn in range(num_comb):
        for dn in range(2):
            lam1_pair = np.zeros([num_dim, num_dim]) if dn == 0 else lam1_n
            redundant = True
            while redundant:
                lam1_n = np.zeros([num_dim, num_dim])
                for j in range(num_dim):
                    if num_latent is None:
                        pacands = np.setdiff1d(np.where(np.sum(lam1_n > 0, axis=0) < num_neighbor)[0], np.where(lam1_pair.T[j, :] != 0)[0])
                        paid = pacands[np.random.permutation(len(pacands))][:num_neighbor] if len(pacands) != 0 else []
                    else:
                        num_observe = num_dim - num_latent
                        pick_interval = int(num_dim / num_observe)
                        obs_idx = np.arange(pick_interval - 1, num_dim, pick_interval)
                        latent_idx = np.setdiff1d(np.arange(num_dim), obs_idx)
                        num_neighbor_obs = int(num_neighbor * num_observe / num_dim)
                        num_neighbor_latent = num_neighbor - num_neighbor_obs
                        if j in obs_idx:
                            lam1_n_j = lam1_n[obs_idx, :]
                        else:
                            lam1_n_j = lam1_n[latent_idx, :]
                        # to observable
                        pacands = obs_idx
                        exclude_num_pa = obs_idx[np.where(np.sum(lam1_n_j[:, obs_idx] > 0, axis=0) >= num_neighbor_obs)[0]]
                        pacands = np.setdiff1d(pacands, exclude_num_pa)
                        pacands = np.setdiff1d(pacands, np.where(lam1_pair.T[j, :] != 0)[0])
                        paid_obs = pacands[np.random.permutation(len(pacands))][:num_neighbor_obs] if len(pacands) != 0 else []
                        # to latent
                        pacands = latent_idx
                        exclude_num_pa = latent_idx[np.where(np.sum(lam1_n_j[:, latent_idx] > 0, axis=0) >= num_neighbor_latent)[0]]
                        pacands = np.setdiff1d(pacands, exclude_num_pa)
                        pacands = np.setdiff1d(pacands, np.where(lam1_pair.T[j, :] != 0)[0])
                        paid_latent = pacands[np.random.permutation(len(pacands))][:num_neighbor_latent] if len(pacands) != 0 else []
                        #
                        paid = np.concatenate([paid_obs, paid_latent]).astype(int)
                    #
                    lam1_n[j, paid] = 1
                if num_latent is not None:
                    obs_idx = np.arange(pick_interval - 1, num_dim, pick_interval)
                    latent_idx = np.setdiff1d(np.arange(num_dim), obs_idx)
                    lam1_n_obs = lam1_n[obs_idx, :][:, obs_idx]
                    lam1_n_latent = lam1_n[latent_idx, :][:, latent_idx]
                if (np.min(np.sum(lam1_n, axis=0)) == np.max(np.sum(lam1_n, axis=0))) and (
                        np.min(np.sum(lam1_n, axis=1)) == np.max(np.sum(lam1_n, axis=1))) and (
                        np.linalg.matrix_rank(lam1_n != 0) >= num_dim - 1) and (
                        np.max(lam1_n + lam1_pair.T) == 1) and (
                        np.linalg.matrix_rank(lam1_n_obs) >= num_observe - 1 if num_latent is not None else True) and (
                        np.linalg.matrix_rank(lam1_n_latent) >= num_latent - 1 if num_latent is not None else True):
                    redundant = False
                    lam1[:, :, dn, cn] = lam1_n
    # scaling
    lam1 = lam1 * np.random.uniform(lam1_range[0], lam1_range[1], size=lam1.shape)

    lam2 = lam1 != 0
    lam2 = lam2 * np.random.uniform(lam2_range[0], lam2_range[1], size=lam2.shape)

    # intra-graph
    lamin1 = np.zeros([num_dim, num_dim, num_group])
    if num_neighbor_in > 0:
        for mn in range(num_group):
            redundant = True
            while redundant:
                lam1_n = np.zeros([num_dim, num_dim])
                for j in range(num_dim):
                    pacands = np.where(np.sum(lam1_n > 0, axis=0) < num_neighbor_in)[0]
                    paid = pacands[np.random.permutation(len(pacands))][:num_neighbor_in] if len(pacands) != 0 else []
                    lam1_n[j, paid] = 1
                if (np.min(np.sum(lam1_n, axis=0)) == np.max(np.sum(lam1_n, axis=0))) and (
                        np.min(np.sum(lam1_n, axis=1)) == np.max(np.sum(lam1_n, axis=1))) and (
                        np.linalg.matrix_rank(lam1_n != 0) >= num_dim - 1) and (
                        np.max(np.abs(np.diag(lam1_n))) == 0) and (
                        np.max(lam1_n + lam1_n.T) == 1):
                    redundant = False
                    lamin1[:, :, mn] = lam1_n
        # scaling
        lamin1 = lamin1 * np.random.uniform(lamin1_range[0], lamin1_range[1], size=lamin1.shape)

    lamin2 = lamin1 != 0
    lamin2 = lamin2 * np.random.uniform(lamin2_range[0], lamin2_range[1], size=lamin2.shape)

    if norm_by_num_parents:
        lam1, lam2, lamin1, lamin2 = norm_lambda_column(lam1, lam2, lamin1, lamin2)

    return lam1, lam2, lamin1, lamin2


# =============================================================
# =============================================================
def gen_net_dag(num_group,
                num_dim,
                lam1_range=None,
                lam2_range=None,
                lamin1_range=None,
                lamin2_range=None,
                num_neighbor=2,
                num_neighbor_in=1,
                num_neighbor_in_max=None,
                norm_by_num_parents=True,
                random_seed=0):
    """Generate graph.
    Args:
        num_group: number of groups
        num_dim: number of variables in a single group
        lam1_range: range of inter-lambda1
        lam2_range: range of inter-lambda2
        lamin1_range: range of intra-lambda1
        lamin2_range: range of intra-lambda2
        num_neighbor: number of neighbors of a variable in a single group-pair
        num_neighbor_in: number of neighbors of a variable in the same group
        num_neighbor_in_max: maximum number of neighbors within each group
        norm_by_num_parents: normalize by number of parents, or not
        random_seed: random seed
    Returns:
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    """

    print("Generating graph (DAG)...")

    if lam1_range is None:
        lam1_range = [1, 1]
    if lam2_range is None:
        lam2_range = [1, 1]
    if lamin1_range is None:
        lamin1_range = [1, 1]
    if lamin2_range is None:
        lamin2_range = [1, 1]
    if num_neighbor_in_max is None:
        num_neighbor_in_max = num_neighbor_in + 1

    # initialize random generator
    np.random.seed(random_seed)

    group_combs = list(combinations(np.arange(num_group), 2))
    num_comb = int(comb(num_group, 2))

    # whole L
    redundant = True
    while redundant:
        lam1 = np.zeros([num_dim * num_group, num_dim * num_group])
        for j in range(num_dim * num_group):
            gidx = j // num_dim
            redundant_j = True
            while redundant_j:
                palist = []
                # intra-group
                for k in range(num_neighbor_in):
                    pacands = np.arange(num_dim * gidx, num_dim * gidx + (j - num_dim * gidx))
                    excludes = np.sum(lam1[pacands, :] != 0, axis=1) >= num_neighbor_in_max
                    pacands = pacands[~excludes]
                    paid = np.random.choice(pacands, 1)[0] if len(pacands) != 0 else []
                    lam1[paid, j] = 1
                    if not (isinstance(paid, list) and len(paid) == 0):
                        palist.append(paid)
                # inter-group
                for g in range(num_group):
                    for k in range(num_neighbor):
                        pacands = np.arange(max(0, (gidx - g - 1) * num_dim), (gidx - g) * num_dim)
                        pacands = np.setdiff1d(pacands, palist)
                        # remove "parents and children" of parents
                        papaids = np.union1d(np.where(np.sum(lam1[:, palist], axis=1) > 0)[0], np.where(np.sum(lam1[palist, :], axis=0) > 0)[0])
                        pacands = np.setdiff1d(pacands, papaids)
                        # # remove parents having many children
                        excludes = np.sum(lam1[pacands, gidx * num_dim:] != 0, axis=1) >= num_neighbor
                        pacands = pacands[~excludes]
                        paid = np.random.choice(pacands, 1)[0] if len(pacands) != 0 else []
                        lam1[paid, j] = 1
                        if not (isinstance(paid, list) and len(paid) == 0):
                            palist.append(paid)
                # check redundancy
                if (gidx > 0) and (j > 0) and (np.min(np.sum(np.abs(lam1[:, :j] - lam1[:, j][:, None]), axis=0)) == 0):
                    lam1[:, j] = 0  # redundant
                else:
                    redundant_j = False
        # check the whole graph is redundant or not
        mat_ranks = []
        num_parent = []
        num_child = []
        for i in range(num_group - 1):
            for j in range(i + 1, num_group):
                lam1_ij = lam1[num_dim * i:num_dim * (i + 1), num_dim * j:num_dim * (j + 1)]
                mat_ranks.append(np.linalg.matrix_rank(lam1_ij != 0))
                num_parent.append(np.sum(lam1_ij != 0, axis=1))
                num_child.append(np.sum(lam1_ij != 0, axis=0))
        if np.min(mat_ranks) == num_dim:
            redundant = False

    # final check
    num_overlap = np.zeros(num_dim * num_group)
    for j in range(num_dim * num_group):
        pa_of_j = np.where(lam1[:, j] != 0)
        pa_of_pa_of_j = np.where(np.sum(lam1[:, pa_of_j], axis=1) > 0)[0]
        intersect = np.intersect1d(pa_of_j, pa_of_pa_of_j)
        num_overlap[j] = len(intersect)
    assert np.min(num_overlap) == 0

    # divide into groups
    lam1mat = lam1
    lam1 = np.zeros([num_dim, num_dim, 2, num_comb])
    for c in range(num_comb):
        a = group_combs[c][0]
        b = group_combs[c][1]
        lam1[:, :, 0, c] = lam1mat[num_dim * a:num_dim * (a + 1), num_dim * b:num_dim * (b + 1)]
        lam1[:, :, 1, c] = lam1mat[num_dim * b:num_dim * (b + 1), num_dim * a:num_dim * (a + 1)]
    lamin1 = np.zeros([num_dim, num_dim, num_group])
    for m in range(num_group):
        lamin1[:, :, m] = lam1mat[num_dim * m: num_dim * (m + 1), num_dim * m: num_dim * (m + 1)]

    # scaling
    lam1 = lam1 * np.random.uniform(lam1_range[0], lam1_range[1], size=lam1.shape)
    lamin1 = lamin1 * np.random.uniform(lamin1_range[0], lamin1_range[1], size=lamin1.shape)
    lam2 = lam1 != 0
    lam2 = lam2 * np.random.uniform(lam2_range[0], lam2_range[1], size=lam2.shape)
    lamin2 = lamin1 != 0
    lamin2 = lamin2 * np.random.uniform(lamin2_range[0], lamin2_range[1], size=lamin2.shape)

    # special graph for the first group to reduce correlation
    num_neighbor_in_g1 = 2
    lamin11 = np.zeros([num_dim, num_dim])
    for j in range(num_neighbor_in_g1 + 1, num_dim):
        pacands = np.setdiff1d(np.arange(j), np.where(np.sum(lamin11 != 0, axis=1) > 0)[0])
        pacands = np.setdiff1d(pacands, np.where(np.sum(lamin11 != 0, axis=0) > 0)[0])
        if len(pacands) >= num_neighbor_in_g1:
            paid = pacands[np.random.permutation(len(pacands))][:num_neighbor_in_g1]
            lamin11[paid, j] = 1
    # scaling
    lamin11 = lamin11 * np.random.uniform(lamin1_range[0], lamin1_range[1], size=lamin11.shape)
    lamin1[:, :, 0] = lamin11

    if norm_by_num_parents:
        lam1, lam2, lamin1, lamin2 = norm_lambda_column(lam1, lam2, lamin1, lamin2)

    return lam1, lam2, lamin1, lamin2


# =============================================================
# =============================================================
def gen_net_sergio(num_group,
                   num_dim,
                   lam1_range=None,
                   lam2_range=None,
                   lamin1_range=None,
                   lamin2_range=None,
                   num_neighbor=2,
                   num_neighbor_in=1,
                   num_latent=None,
                   num_neighbor_in_max=None,
                   random_seed=0):
    """Generate graph.
    Args:
        num_group: number of groups
        num_dim: number of variables in a single group
        lam1_range: range of inter-lambda1
        lam2_range: range of inter-lambda2
        lamin1_range: range of intra-lambda1
        lamin2_range: range of intra-lambda2
        num_neighbor: number of neighbors of a variable in a single group-pair
        num_neighbor_in: number of neighbors of a variable in the same group
        num_latent: number of latent confounders (included in num_dim)
        num_neighbor_in_max: maximum number of neighbors within each group
        random_seed: random seed
    Returns:
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    """

    print("Generating graph (SERGIO)...")

    if lam1_range is None:
        lam1_range = [1, 1]
    if lam2_range is None:
        lam2_range = [1, 1]
    if lamin1_range is None:
        lamin1_range = [1, 1]
    if lamin2_range is None:
        lamin2_range = [1, 1]
    if num_neighbor_in_max is None:
        num_neighbor_in_max = num_neighbor_in + 1

    # initialize random generator
    np.random.seed(random_seed)

    group_combs = list(combinations(np.arange(num_group), 2))
    num_comb = int(comb(num_group, 2))

    # inter-group
    lam1 = np.zeros([num_dim, num_dim, 2, num_comb])
    for cn in range(num_comb):
        lam1_pair = np.zeros([num_dim, num_dim])
        redundant = True
        while redundant:
            lam1_n = np.zeros([num_dim, num_dim])
            for j in range(num_dim):
                if num_latent is None:
                    pacands = np.setdiff1d(np.where(np.sum(lam1_n > 0, axis=0) < num_neighbor)[0], np.where(lam1_pair.T[j, :] != 0)[0])
                    paid = pacands[np.random.permutation(len(pacands))][:num_neighbor] if len(pacands) != 0 else []
                else:
                    num_observe = num_dim - num_latent
                    pick_interval = int(num_dim / num_observe)
                    obs_idx = np.arange(pick_interval - 1, num_dim, pick_interval)
                    latent_idx = np.setdiff1d(np.arange(num_dim), obs_idx)
                    num_neighbor_obs = int(num_neighbor * num_observe / num_dim)
                    num_neighbor_latent = num_neighbor - num_neighbor_obs
                    if j in obs_idx:
                        lam1_n_j = lam1_n[obs_idx, :]
                    else:
                        lam1_n_j = lam1_n[latent_idx, :]
                    # to observable
                    pacands = obs_idx
                    exclude_num_pa = obs_idx[np.where(np.sum(lam1_n_j[:, obs_idx] > 0, axis=0) >= num_neighbor_obs)[0]]
                    pacands = np.setdiff1d(pacands, exclude_num_pa)
                    pacands = np.setdiff1d(pacands, np.where(lam1_pair.T[j, :] != 0)[0])
                    paid_obs = pacands[np.random.permutation(len(pacands))][:num_neighbor_obs] if len(pacands) != 0 else []
                    # to latent
                    pacands = latent_idx
                    exclude_num_pa = latent_idx[np.where(np.sum(lam1_n_j[:, latent_idx] > 0, axis=0) >= num_neighbor_latent)[0]]
                    pacands = np.setdiff1d(pacands, exclude_num_pa)
                    pacands = np.setdiff1d(pacands, np.where(lam1_pair.T[j, :] != 0)[0])
                    paid_latent = pacands[np.random.permutation(len(pacands))][:num_neighbor_latent] if len(pacands) != 0 else []
                    #
                    paid = np.concatenate([paid_obs, paid_latent]).astype(int)
                #
                lam1_n[j, paid] = 1
            if num_latent is not None:
                obs_idx = np.arange(pick_interval - 1, num_dim, pick_interval)
                latent_idx = np.setdiff1d(np.arange(num_dim), obs_idx)
                lam1_n_obs = lam1_n[obs_idx, :][:, obs_idx]
                lam1_n_latent = lam1_n[latent_idx, :][:, latent_idx]
            if (np.min(np.sum(lam1_n, axis=0)) == np.max(np.sum(lam1_n, axis=0))) and (
                    np.min(np.sum(lam1_n, axis=1)) == np.max(np.sum(lam1_n, axis=1))) and (
                    np.linalg.matrix_rank(lam1_n != 0) >= num_dim - 1) and (
                    np.max(lam1_n + lam1_pair.T) == 1) and (
                    np.linalg.matrix_rank(lam1_n_obs) >= num_observe - 1 if num_latent is not None else True) and (
                    np.linalg.matrix_rank(lam1_n_latent) >= num_latent - 1 if num_latent is not None else True):
                redundant = False
                lam1[:, :, 0, cn] = lam1_n
    # scaling
    lam1 = lam1 * np.random.uniform(lam1_range[0], lam1_range[1], size=lam1.shape)

    # intra-group
    lamin1 = np.zeros([num_dim, num_dim, num_group])
    if num_neighbor_in > 0:
        # special graph for the first group for reduce correlation
        num_neighbor_in_g1 = 2
        lamin11 = np.zeros([num_dim, num_dim])
        for j in range(num_neighbor_in_g1 + 1, num_dim):
            pacands = np.setdiff1d(np.arange(j), np.where(np.sum(lamin11 != 0, axis=1) > 0)[0])
            pacands = np.setdiff1d(pacands, np.where(np.sum(lamin11 != 0, axis=0) > 0)[0])
            if len(pacands) >= num_neighbor_in_g1:
                paid = pacands[np.random.permutation(len(pacands))][:num_neighbor_in_g1]
                lamin11[paid, j] = 1
        lamin1[:, :, 0] = lamin11

        for mn in range(1, num_group):
            lam1_n = np.zeros([num_dim, num_dim])
            for j in range(1, num_dim):
                for k in range(num_neighbor_in):
                    pacands = np.arange(j)
                    excludes = np.sum(lam1_n[pacands, :] != 0, axis=1) >= num_neighbor_in_max
                    pacands = pacands[~excludes]
                    paid = np.random.choice(pacands, 1)[0] if len(pacands) != 0 else []
                    lam1_n[paid, j] = 1
            lamin1[:, :, mn] = lam1_n

        # scaling
        lamin1 = lamin1 * np.random.uniform(lamin1_range[0], lamin1_range[1], size=lamin1.shape)

    # additional procedure
    lam1mat = wg_to_w(wg=lam1, num_group=num_group, wgin=lamin1)

    # make the last variable of each group as leaves
    for g in range(num_group - 1):
        lam1mat[num_dim * (g + 1) - 1, :] = 0
        lam1mat[:, num_dim * (g + 1) - 1] = 0

    leaf_var = np.where(np.sum(lam1mat != 0, axis=1) == 0)[0]
    leaf_var = leaf_var[leaf_var < num_dim * (num_group - 1)]
    max_leaf_num = int(np.ceil(num_dim / len(leaf_var)))
    if num_latent is not None:
        # assume that num_dim == num_latent
        assert num_latent == int(num_dim / 2)
        max_leaf_num = int(max_leaf_num / 2)
    for j in range(num_dim * (num_group - 1), num_dim * num_group):
        if num_latent is not None:
            if np.mod(j, 2) == 0:
                continue
        leaf_var_cand = leaf_var[np.sum(lam1mat[:, leaf_var] != 0, axis=0) < max_leaf_num]
        leaf_var_j = leaf_var_cand[np.random.permutation(len(leaf_var_cand))][:1]
        lam1mat[j, leaf_var_j] = lam1_range[1]

    # assign signs half-half
    for j in range(lam1mat.shape[1]):
        pa_ids = np.where(lam1mat[:, j] != 0)[0]
        if len(pa_ids) > 0:
            num_pos = int(np.ceil(len(pa_ids) / 2))
            num_neg = len(pa_ids) - num_pos
            neg_idx = np.random.permutation(pa_ids)[:num_neg]
            lam1mat[neg_idx, j] = - lam1mat[neg_idx, j]

    lam1, lamin1 = w_to_wg(lam1mat, num_group=num_group)

    lam2 = lam1 != 0
    lam2 = lam2 * np.random.uniform(lam2_range[0], lam2_range[1], size=lam2.shape)

    lamin2 = lamin1 != 0
    lamin2 = lamin2 * np.random.uniform(lamin2_range[0], lamin2_range[1], size=lamin2.shape)

    return lam1, lam2, lamin1, lamin2


# =============================================================
# =============================================================
def norm_lambda_column(lam1,
                       lam2,
                       lamin1,
                       lamin2):
    """Normalize adjacency matrix by the number of parents.
    Args:
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    Returns:
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
    """

    num_dim, _, num_group = lamin1.shape
    group_combs = list(combinations(np.arange(num_group), 2))
    num_comb = int(comb(num_group, 2))

    # count the number of parents
    num_parents = np.zeros([num_dim, num_group])
    for m in range(num_group):
        num_parents[:, m] = num_parents[:, m] + np.sum(lamin1[:, :, m] > 0, axis=0)  # parents in the same group
        for c in range(num_comb):
            if np.where(np.array(group_combs[c]) == m)[0].size > 0:
                m_side = np.where(np.array(group_combs[c]) == m)[0][0]  # o=left, 1=right
                opp_side = 1 if m_side == 0 else 0  # o=left, 1=right
                num_parents[:, m] = num_parents[:, m] + np.sum(lam1[:, :, opp_side, c] > 0, axis=0)
    num_parents_min1 = num_parents.copy()
    num_parents_min1[num_parents_min1 == 0] = 1
    # adjust the weights by the number of parents
    lamin1 = lamin1 / num_parents_min1[None, :, :]
    lamin2 = lamin2 * num_parents_min1[None, :, :]
    for c in range(num_comb):
        lam1[:, :, 0, c] = lam1[:, :, 0, c] / num_parents_min1[None, :, group_combs[c][1]]
        lam2[:, :, 0, c] = lam2[:, :, 0, c] * num_parents_min1[None, :, group_combs[c][1]]
        lam1[:, :, 1, c] = lam1[:, :, 1, c] / num_parents_min1[None, :, group_combs[c][0]]
        lam2[:, :, 1, c] = lam2[:, :, 1, c] * num_parents_min1[None, :, group_combs[c][0]]

    return lam1, lam2, lamin1, lamin2


# =============================================================
# =============================================================
def gen_s(num_data,
          lam1,
          lam2,
          lamin1,
          lamin2,
          lam1_mean,
          ar_alpha=1,
          num_rep=6,
          dag=False,
          nonlinearity='ReLU',
          random_seed=0):
    """Generate latent variables
    Args:
        num_data: number of data
        lam1: lambda1 [dim, dim, 2, group-pair]
        lam2: lambda2 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lamin2: lambda_in2 [dim, dim, group]
        lam1_mean: average value of lam1
        ar_alpha: AR parameter
        num_rep: number of repetition of sampling for Gibbs sampling
        dag: DAG(True) or cyclic(False)
        nonlinearity: nonlinearity of cross potential {'ReLU', 'tanh'}
        random_seed: random seed
    Returns:
        s: latent variables [data, group, dim]
    """

    # initialize random generator
    np.random.seed(random_seed)

    num_dim, _, _, num_comb = lam1.shape
    num_group = lamin1.shape[-1]
    group_combs = list(combinations(np.arange(num_group), 2))

    assert nonlinearity in {'ReLU', 'tanh'}

    s = np.zeros([num_data, num_group, num_dim])
    siglist = np.zeros([num_data, num_group, num_dim])
    mulist = np.zeros([num_data, num_group, num_dim])
    sigposlist = np.zeros([num_data, num_group, num_dim])
    muposlist = np.zeros([num_data, num_group, num_dim])
    if dag:
        for m in range(num_group):
            sys.stdout.write('\rGenerating s... %d/%d' % (m + 1, num_group))
            sys.stdout.flush()
            for n in range(num_dim):
                a1 = np.zeros(num_data)
                a2 = np.zeros(num_data)
                # intra-group
                lamin1_to_n = lamin1[:, n, m]
                lamin2_to_n = lamin2[:, n, m]
                a1 = a1 + np.sum(lamin1_to_n)
                if nonlinearity == 'ReLU':
                    s_m_nonlin = s[:, m, :].copy()
                    s_m_nonlin[s_m_nonlin < 0] = 0
                elif nonlinearity == 'tanh':
                    s_m_nonlin = np.tanh(s[:, m, :])
                a2 = a2 + np.sum(2 * ar_alpha * lamin1_to_n[None, :] * lamin2_to_n[None, :] * s_m_nonlin, axis=1)
                # inter-groups
                for c in range(num_comb):
                    if np.where(np.array(group_combs[c]) == m)[0].size > 0:
                        m_side = np.where(np.array(group_combs[c]) == m)[0][0]  # o=left, 1=right
                        opp_side = 1 if m_side == 0 else 0  # o=left, 1=right
                        m_opp = group_combs[c][opp_side]
                        # from parents on the opposite side
                        lam1_to_n = lam1[:, n, opp_side, c]
                        lam2_to_n = lam2[:, n, opp_side, c]
                        #
                        a1 = a1 + np.sum(lam1_to_n)
                        if nonlinearity == 'ReLU':
                            s_opp_nonlin = s[:, m_opp, :].copy()
                            s_opp_nonlin[s_opp_nonlin < 0] = 0
                        elif nonlinearity == 'tanh':
                            s_opp_nonlin = np.tanh(s[:, m_opp, :])
                        a2 = a2 + np.sum(2 * ar_alpha * lam1_to_n[None, :] * lam2_to_n[None, :] * s_opp_nonlin, axis=1)

                a1[a1 == 0] = lam1_mean
                sig = 1 / np.sqrt(2 * a1)
                mu = - a2 / (2 * a1)
                sn = np.random.normal(mu, sig)

                s[:, m, n] = sn
                siglist[:, m, n] = sig
                mulist[:, m, n] = mu

    else:  # cyclic
        for i in range(num_rep):
            # generate for each variable
            for n in range(num_dim):
                sys.stdout.write('\rGenerating s... %d/%d, %d/%d' % (i + 1, num_rep, n + 1, num_dim))
                sys.stdout.flush()
                if i == num_rep - 1:
                    update_flag = np.mod(np.arange(num_data), num_dim) >= n
                else:
                    update_flag = np.ones(num_data, dtype=bool)
                for m in range(num_group):
                    a1 = np.zeros(np.sum(update_flag))
                    a2 = np.zeros(np.sum(update_flag))
                    a1pos = np.zeros(np.sum(update_flag))
                    a2pos = np.zeros(np.sum(update_flag))
                    for c in range(num_comb):
                        if np.where(np.array(group_combs[c]) == m)[0].size > 0:
                            m_side = np.where(np.array(group_combs[c]) == m)[0][0]  # o=left, 1=right
                            opp_side = 1 if m_side == 0 else 0  # o=left, 1=right
                            m_opp = group_combs[c][opp_side]
                            # from parents on the opposite side
                            lam1_to_n = lam1[:, n, opp_side, c]
                            lam2_to_n = lam2[:, n, opp_side, c]
                            lamin1_to_n = lamin1[:, n, m]
                            lamin2_to_n = lamin2[:, n, m]
                            a1 = a1 + np.sum(lam1_to_n) + np.sum(lamin1_to_n)
                            if nonlinearity == 'ReLU':
                                s_opp_relu = s[update_flag, m_opp, :].copy()
                                s_opp_relu[s_opp_relu < 0] = 0
                                s_m_relu = s[update_flag, m, :].copy()
                                s_m_relu[s_m_relu < 0] = 0
                                a2 = a2 + np.sum(2 * ar_alpha * lam1_to_n[None, :] * lam2_to_n[None, :] * s_opp_relu, axis=1) + \
                                     np.sum(2 * ar_alpha * lamin1_to_n[None, :] * lamin2_to_n[None, :] * s_m_relu, axis=1)
                            # from children
                            lam1_from_n = lam1[n, :, m_side, c]
                            lam2_from_n = lam2[n, :, m_side, c]
                            lamin1_from_n = lamin1[n, :, m]
                            lamin2_from_n = lamin2[n, :, m]
                            a1pos = a1pos + np.sum(lam1_from_n * (ar_alpha * lam2_from_n)**2) + np.sum(lamin1_from_n * (ar_alpha * lamin2_from_n)**2)
                            if nonlinearity == 'ReLU':
                                s_opp_from_n = s[update_flag, m_opp, :].copy()
                                s_m_from_n = s[update_flag, m, :].copy()
                                a2pos = a2pos + np.sum(2 * ar_alpha * lam1_from_n[None, :] * lam2_from_n[None, :] * s_opp_from_n, axis=1) + \
                                        np.sum(2 * ar_alpha * lamin1_from_n[None, :] * lamin2_from_n[None, :] * s_m_from_n, axis=1)

                    a1[a1 == 0] = lam1_mean
                    sig = 1 / np.sqrt(2 * a1)
                    mu = - a2 / (2 * a1)
                    sigpos = 1 / np.sqrt(2 * (a1 + a1pos))
                    mupos = - (a2 + a2pos) / (2 * (a1 + a1pos))
                    sn = rand_normal_np(mu, sig, mupos, sigpos)

                    s[update_flag, m, n] = sn
                    siglist[update_flag, m, n] = sig
                    mulist[update_flag, m, n] = mu
                    sigposlist[update_flag, m, n] = sigpos
                    muposlist[update_flag, m, n] = mupos

    sys.stdout.write('\r\n')

    return s


# =============================================================
# =============================================================
def rand_normal_np(loc_n, scale_n, loc_p, scale_p):
    """Sample from Gaussian distribution with different parameters for negative and positive sides
    Args:
        loc_n: location parameter of negative side
        scale_n: scale parameter of negative side
        loc_p: location parameter of positive side
        scale_p: scale parameter of positive side
    Returns:
        x: samples
    """

    Z_n = scale_n * norm.cdf(np.zeros_like(loc_n), loc_n, scale_n)
    Z_p = scale_p * (1 - norm.cdf(np.zeros_like(loc_p), loc_p, scale_p))
    pos_flag = np.random.uniform(size=loc_n.shape) > Z_n / (Z_n + Z_p)

    # negative side
    xn = np.zeros(np.sum(~pos_flag))
    loc = loc_n[~pos_flag]
    scale = scale_n[~pos_flag]
    while np.sum(xn == 0) > 0:
        gen_idx = xn == 0
        xni = np.random.normal(loc[gen_idx], scale[gen_idx])
        xni[xni > 0] = 0
        xn[gen_idx] = xni

    # positive side
    xp = np.zeros(np.sum(pos_flag))
    loc = loc_p[pos_flag]
    scale = scale_p[pos_flag]
    while np.sum(xp == 0) > 0:
        gen_idx = xp == 0
        xni = np.random.normal(loc[gen_idx], scale[gen_idx])
        xni[xni < 0] = 0
        xp[gen_idx] = xni

    x = np.zeros_like(loc_n)
    x[~pos_flag] = xn
    x[pos_flag] = xp

    return x


# =============================================================
# =============================================================
def gen_s_lap(num_data,
              lam1,
              lamin1,
              lam1_mean,
              ar_alpha=1,
              ar_beta=1,
              dag=False,
              nonlinearity='tanh',
              random_seed=0):
    """Generate latent variables and observations
    Args:
        num_data: number of data
        lam1: lambda1 [dim, dim, 2, group-pair]
        lamin1: lambda_in1 [dim, dim, group]
        lam1_mean: average value of lam1
        ar_alpha: AR parameter
        ar_beta: AR parameter
        dag: DAG(True) or cyclic(False)
        nonlinearity: nonlinearity of cross potential {'tanh'}
        random_seed: random seed
    Returns:
        s: latent variables [data, group, dim]
    """

    # initialize random generator
    np.random.seed(random_seed)

    num_dim, _, _, num_comb = lam1.shape
    num_group = lamin1.shape[-1]
    group_combs = list(combinations(np.arange(num_group), 2))

    assert nonlinearity in {'tanh'}

    s = np.zeros([num_data, num_group, num_dim])
    if dag:
        for m in range(num_group):
            sys.stdout.write('\rGenerating s... %d/%d' % (m + 1, num_group))
            sys.stdout.flush()
            for n in range(num_dim):
                an = np.zeros([0])
                sn = np.zeros([num_data, 0])
                # intra-group
                lamin1_to_n = lamin1[:, n, m]
                an = np.concatenate([an, lamin1_to_n[lamin1_to_n != 0]])
                sn = np.concatenate([sn, s[:, m, lamin1_to_n != 0]], axis=1)
                # inter-group
                for c in range(num_comb):
                    if np.where(np.array(group_combs[c]) == m)[0].size > 0:
                        m_side = np.where(np.array(group_combs[c]) == m)[0][0]  # o=left, 1=right
                        opp_side = 1 if m_side == 0 else 0  # o=left, 1=right
                        m_opp = group_combs[c][opp_side]
                        lam1_to_n = lam1[:, n, opp_side, c]
                        s_to_n = s[:, m_opp, :].copy()
                        an = np.concatenate([an, lam1_to_n[lam1_to_n != 0]])
                        sn = np.concatenate([sn, s_to_n[:, lam1_to_n != 0]], axis=1)

                if np.sum(an != 0) == 0:
                    an = np.concatenate([an, lam1_mean.reshape(-1)])
                    sn = np.concatenate([sn, np.zeros([num_data, 1])], axis=1)

                an = np.tile(an[None, :], (num_data, 1))
                sn_nonlin = np.tanh(ar_beta * sn)
                alpha_sn = - ar_alpha * sn_nonlin

                # generate
                sort_idx = np.argsort(alpha_sn, axis=1)
                alpha_sn_sorted = np.sort(alpha_sn, axis=1)
                an_sorted = an[np.arange(alpha_sn.shape[0])[:, None], sort_idx]
                sn = random_piecewise_lap(an_sorted, alpha_sn_sorted, size=1)

                s[:, m, n] = sn.reshape(-1)

    else:  # cyclic
        raise ValueError

    sys.stdout.write('\r\n')

    return s


# =============================================================
# =============================================================
def random_piecewise_lap(A, alphas, size, b_max=7):
    """Sample from piecewise Laplace distribution
    Args:
        A: [data, anker]
        alphas: alpha * s, [data, anker]
        size: sample size
        b_max:
        (Note) alphas is supposed to be sorted for each column, A and znorm as well accordingly
    Returns:
        x: samples [data, size]
    """

    num_data, num_anker = A.shape

    # calculate segment-wise partion function
    z = np.zeros([num_data, num_anker + 1])
    for k in range(num_anker - 1):
        Ak = A.copy()
        Ak[:, :(k + 1)] = - Ak[:, :(k + 1)]
        ak0flag = np.sum(Ak, axis=1) == 0
        # sum(Ak) > 0
        zkp1 = np.exp(np.sum(Ak * (alphas[:, k + 1][:, None] - alphas), axis=1))
        zkp0 = np.exp(np.sum(Ak * (alphas[:, k][:, None] - alphas), axis=1))
        zk = (zkp1 - zkp0) / np.sum(Ak, axis=1)
        # sum(Ak) = 0
        zka0 = np.exp(np.sum(- Ak * alphas, axis=1)) * (alphas[:, k + 1] - alphas[:, k])
        #
        z[~ak0flag, k + 1] = zk[~ak0flag]
        z[ak0flag, k + 1] = zka0[ak0flag]
    zlinf = np.exp(np.sum(A * (alphas[:, 0][:, None] - alphas), axis=1)) / np.sum(A, axis=1)
    zrinf = np.exp(np.sum(A * (alphas - alphas[:, -1][:, None]), axis=1)) / np.sum(A, axis=1)
    z[:, 0] = zlinf
    z[:, -1] = zrinf
    znorm = z / np.sum(z, axis=1, keepdims=True)  # [data, dim + 1]
    zcum = np.concatenate([np.zeros([num_data, 1]), np.cumsum(znorm, axis=1)], axis=1)
    # [data, dim + 2] cumulative sum at the right border of each segment

    x = np.zeros([num_data, size])
    for n in range(size):
        randval = np.random.uniform(0, 1, [num_data, 1])
        k = np.argmax((zcum[:, :-1] <= randval) & (zcum[:, 1:] > randval), axis=1)

        sign = 1 - 2 * (np.tile(np.arange(A.shape[1])[None, :], (num_data, 1)) >= k[:, None]).astype(float)
        Ak = A * sign

        ak = np.sum(Ak, axis=1)  # [dim]
        bk = - np.sum(Ak * alphas, axis=1)  # [dim]
        kwmat = np.zeros_like(znorm)  # [node + 1, dim]
        kwmat[:, 1:-1] = alphas[:, 1:] - alphas[:, :-1]
        kwmat[:, [0, -1]] = np.inf
        kwidth = kwmat[np.arange(num_data), k]

        kl = np.zeros(num_data)
        kr = np.zeros(num_data)
        kl[k > 0] = alphas[k > 0][np.arange(np.sum(k > 0)), k[k > 0] - 1]
        kl[k <= 0] = -np.inf
        kr[k < num_anker] = alphas[k < num_anker][np.arange(np.sum(k < num_anker)), k[k < num_anker]]
        kr[k >= num_anker] = np.inf

        # sampling
        xn = np.zeros(num_data)

        # a > 0
        in_flag = ak > 0
        bias = - bk[in_flag] / ak[in_flag]
        lower = kl[in_flag] - bias
        upper = kr[in_flag] - bias
        if b_max is not None:
            upper[upper - lower > b_max] = lower[upper - lower > b_max] + b_max
        scale = 1/np.abs(ak[in_flag])
        xn[in_flag] = truncexpon.rvs(b=(upper - lower) / scale, loc=lower, scale=scale) + bias

        # a < 0
        in_flag = ak < 0
        bias = bk[in_flag] / ak[in_flag]
        lower = bias - kr[in_flag]
        upper = bias - kl[in_flag]
        if b_max is not None:
            upper[upper - lower > b_max] = lower[upper - lower > b_max] + b_max
        scale = 1/np.abs(ak[in_flag])
        xn[in_flag] = - truncexpon.rvs(b=(upper - lower) / scale, loc=lower, scale=scale) + bias

        # a == 0
        in_flag = ak == 0
        xn[in_flag] = np.random.uniform(kl[in_flag], kr[in_flag])

        x[:, n] = xn

    return x


# =============================================================
# =============================================================
def gen_mlp(num_dim,
            num_layer,
            iter_cond_thresh=10000,
            cond_thresh_ratio=0.25,
            negative_slope=0.2,
            random_seed=0):
    """Generate MLP
    Args:
        num_dim: number of variables
        num_layer: number of layers
        iter_cond_thresh: number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: percentile of condition number to decide its threshold
        negative_slope: negative slope of leakyReLU (for properly scaling weights)
        random_seed: random seed
    Returns:
        mixlayer: parameters of mixing layers
    """

    print("Generating MLP...")

    # initialize random generator
    np.random.seed(random_seed)

    # generate W
    def genw(num_in, num_out, nonlin=True):
        wf = np.random.uniform(-1, 1, [num_out, num_in])
        if nonlin:
            wf = wf * np.sqrt(6/((1 + negative_slope**2)*num_in))
        else:
            wf = wf * np.sqrt(6/(num_in*2))
        return wf

    # determine threshold of cond
    condlist = np.zeros([iter_cond_thresh])
    for i in range(iter_cond_thresh):
        w = genw(num_dim, num_dim)
        condlist[i] = np.linalg.cond(w)
    condlist.sort()
    cond_thresh = condlist[int(iter_cond_thresh * cond_thresh_ratio)]
    print("    cond thresh: {0:f}".format(cond_thresh))

    mixlayer = []
    for ln in range(num_layer):
        condw = cond_thresh + 1
        while condw > cond_thresh:
            if ln == 0:  # 1st layer
                w = genw(num_dim, num_dim, nonlin=(num_layer != 1))
            elif ln == num_layer-1:  # last layer
                w = genw(num_dim, num_dim, nonlin=False)
            else:
                w = genw(num_dim, num_dim, nonlin=True)
            condw = np.linalg.cond(w)
        print("    L{0:d}: cond={1:f}".format(ln, condw))
        b = np.zeros(w.shape[-1]).reshape([-1, 1])
        mixlayer.append({"W": w.copy(), "b": b.copy()})

    return mixlayer


# =============================================================
# =============================================================
def gen_x(s,
          mlplayer,
          negative_slope=None):
    """Generate observations
    Args:
        s: latent variables [data, group, dim]
        mlplayer: parameters of mixing layers (gen_mlp)
        negative_slope: negative slope of leaky ReLU
    Returns:
        x: observations [data, group, dim]
    """

    num_data, num_group, num_dim = s.shape

    # apply MLP
    if (len(mlplayer) > 0) and (len(mlplayer[0]) > 0):
        x = np.zeros_like(s)
        for m in range(num_group):
            x[:, m, :] = apply_mlp(s[:, m, :], mlplayer[m], negative_slope=negative_slope)
    else:
        x = s.copy()

    return x


# =============================================================
# =============================================================
def apply_mlp(s,
              mlplayer,
              nonlinear_type='ReLU',
              negative_slope=0.2):
    """Apply MLP to latent variables
    Args:
        s: input signals [data, dim]
        mlplayer: parameters of MLP generated by gen_mlp_parms
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
    Returns:
        x: mixed signals [data, dim]
    """

    num_layer = len(mlplayer)

    x = s.copy()
    for ln in range(num_layer):
        x = x + mlplayer[ln]['b'].reshape([1, -1])
        x = np.dot(x, mlplayer[ln]['W'].T)
        if ln != num_layer - 1:  # no nolinearity for the last layer
            if nonlinear_type == "ReLU":  # leaky-ReLU
                x[x < 0] = negative_slope * x[x < 0]
            else:
                raise ValueError

    return x


# =============================================================
# =============================================================
def w_to_wg(w, num_group=None, num_dim=None):
    """ W to Wg
    Args:
        w: [dim * group, dim * group]
        num_group:
        num_dim:
    Returns:
        wg: [dim, dim, 2, group-pair]
        wgin: [dim, dim, group]
    """

    if num_group is not None:
        num_dim = int(w.shape[0] / num_group)
    elif num_dim is not None:
        num_group = int(w.shape[0] / num_dim)
    else:
        raise ValueError

    combs = list(combinations(np.arange(num_group), 2))
    num_comb = len(combs)

    wg = np.zeros([num_dim, num_dim, 2, num_comb])
    for c in range(num_comb):
        a = combs[c][0]
        b = combs[c][1]
        wg[:, :, 0, c] = w[num_dim * a:num_dim * (a + 1), num_dim * b:num_dim * (b + 1)]
        wg[:, :, 1, c] = w[num_dim * b:num_dim * (b + 1), num_dim * a:num_dim * (a + 1)]

    wgin = np.zeros([num_dim, num_dim, num_group])
    for m in range(num_group):
        wgin[:, :, m] = w[num_dim * m:num_dim * (m + 1), num_dim * m:num_dim * (m + 1)]

    return wg, wgin


# =============================================================
# =============================================================
def wg_to_w(wg, num_group, wgin=None):
    """ Wg to W
    Args:
        wg: [dim, dim, 2, group-pair]
        num_group:
        wgin: [dim, dim, group]
    Returns:
        w: [dim * group, dim * group]
    """

    num_dim, _, _, num_comb = wg.shape
    combs = list(combinations(np.arange(num_group), 2))

    w = np.zeros([num_dim * num_group, num_dim * num_group])
    w[:] = np.nan
    for c in range(num_comb):
        a = combs[c][0]
        b = combs[c][1]
        w[num_dim * a:num_dim * (a + 1), num_dim * b:num_dim * (b + 1)] = wg[:, :, 0, c]
        w[num_dim * b:num_dim * (b + 1), num_dim * a:num_dim * (a + 1)] = wg[:, :, 1, c]

    if wgin is not None:
        for m in range(num_group):
            w[num_dim * m:num_dim * (m + 1), num_dim * m:num_dim * (m + 1)] = wgin[:, :, m]

    return w


# =============================================================
# =============================================================
def simulate_sergio(G, n_samples, hill=2, mr_range: tuple = (0.5, 2.0), seed=0):
    """ Wg to W
    Args:
        G: graph [dim, dim]
        n_samples: number of samples
        hill: hill coefficient
        mr_range: MR range
        seed: random seed
    Returns:
        expr: gene expression
    """
    def write_rows(path, rows):
        file = open(path, 'w')
        for row in rows:
            line = ''
            for val in row:
                line += ', ' + str(val)
            line = line[2:] + '\n'
            file.write(line)

    swap_dir = './storage/'
    if not os.path.isdir(swap_dir):
        os.makedirs(swap_dir)

    nodes = np.array(list(G.nodes))
    indegree = np.array(list(dict(G.in_degree(nodes)).values()))
    MRs = nodes[indegree == 0]
    targets = nodes[indegree != 0]
    mr_rows = [[mr] + list(np.random.uniform(mr_range[0], mr_range[1], n_samples)) for mr in MRs]
    grn_rows = []
    for target in targets:
        parents = list(G.predecessors(target))
        weights = [G[parent][target]['weight'] for parent in parents]
        n_hill = [hill] * len(weights)
        row = [target, len(parents)] + parents + weights + n_hill
        grn_rows.append(row)

    mr_path = swap_dir + 'MR%d.txt' % seed
    grn_path = swap_dir + 'GRN%d.txt' % seed
    write_rows(mr_path, mr_rows)
    write_rows(grn_path, grn_rows)

    sim = sergio(number_genes=len(nodes), number_bins=n_samples, number_sc=1, noise_params=1, decays=0.8,
                 sampling_state=15, noise_type='dpd')
    sim.build_graph(input_file_taregts=grn_path, input_file_regs=mr_path)
    sim.simulate()
    expr = sim.getExpressions()
    expr = np.concatenate(expr, axis=1)
    return expr

