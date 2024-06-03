""" Training
    Main script for training the model
"""


import os
import pickle
import shutil
import tarfile
from sklearn.decomposition import PCA

from subfunc.generate_dataset import generate_dataset
from gcarl.train import train
from gcarl_3dident.train import train as train_3dident
from subfunc.showdata import *

# parameters ==================================================
# =============================================================

# data generation ---------------------------------------------
# common
num_layer = 3  # number of layers of mixing-MLP f
num_group = 3  # number of groups
num_neighbor = 2  # number of neighbors of a variable in a single group-pair
num_neighbor_in = 1  # number of neighbors of a variable within a group
lam2 = [1, 1]
lamin2 = lam2
random_seed = 0  # random seed

# simulation1
dist_type = 'Laplace'  # noise distribution
dag = True  # DAG or not
num_dim = 10  # number of variables in a group
num_data = 2**16  # number of samples
num_latent = None  # number of latent confounders
ar_alpha = 3  # AR parameter
ar_beta = 0.8  # AR parameter
lam1 = [0.9, 1]  # modulation range of inter-L
lamin1 = lam1  # modulation range of intra-L
phi_type = 'lap-mlp'  # model type
phi_share = True  # share phi across group-pairs or not
apply_pca = True  # apply PCA for preprocessing or not

# # simulation2
# dist_type = 'Gauss'  # noise distribution
# dag = False  # DAG or not
# num_dim = 20  # number of variables in a group
# num_data = 2**20  # number of samples
# num_latent = int(num_dim / 2)  # number of latent confounders
# ar_alpha = 0.3  # AR parameter
# ar_beta = None  # AR parameter
# lam1 = [0.9, 1]  # modulation range of inter-L
# lamin1 = lam1  # modulation range of intra-L
# phi_type = 'gauss-maxout'  # model type
# phi_share = False  # share phi across group-pairs or not
# apply_pca = True  # apply PCA for preprocessing or not

# # GRN
# dist_type = 'GRN'  # noise distribution
# dag = True  # DAG or not
# num_dim = 20  # number of variables in a group
# num_data = 2**18  # number of samples
# num_latent = int(num_dim / 2)  # number of latent confounders
# ar_alpha = None  # AR parameter
# ar_beta = None  # AR parameter
# lam1 = [0.25, 0.25]  # modulation range of inter-L
# lamin1 = lam1  # modulation range of intra-L
# phi_type = 'gauss-mlp'  # model type
# phi_share = True  # share phi across group-pairs or not
# apply_pca = True  # apply PCA for preprocessing or not

# # 3dIdent high-dimensional image
# dist_type = 'Gauss'  # noise distribution
# dag = False  # DAG or not
# num_dim = 20  # number of variables in a group
# num_data = 2**20  # number of samples
# num_latent = int(num_dim / 2)  # number of latent confounders
# ar_alpha = 0.3  # AR parameter
# ar_beta = None  # AR parameter
# lam1 = [0.9, 1]  # modulation range of inter-L
# lamin1 = lam1  # modulation range of intra-L
# phi_type = 'gauss-maxout'  # model type
# phi_share = False  # share phi across group-pairs or not
# apply_pca = False  # apply PCA for preprocessing or not
# num_layer = 0  # observational mixings are generated within train.py
# data_path = './data/3dident/train'  # path to the data folder (https://zenodo.org/records/4502485#.YgWm1fXMKbg)


# MLP ---------------------------------------------------------
num_layer_z = 2  # number of layers of hz (\bar\psi)
num_hdim = num_dim if num_latent is None else num_dim - num_latent
num_h_nodes = [2 * num_hdim] * (num_layer - 1) + [num_hdim]  # h
num_hz_nodes = [2 * num_hdim] * (num_layer_z - 1) + [num_hdim]  # \bar\psi
num_hp_nodes = [5, 1]  # \psi
# list of the number of nodes of each hidden layer of h-MLP
# [layer1, layer2, ..., layer(num_layer)]

# training ----------------------------------------------------
initial_learning_rate = 0.1  # initial learning rate
momentum = 0.9  # momentum parameter of SGD
max_steps = int(1e6)  # number of iterations (mini-batches)
decay_steps = [int(6e5), int(9e5)]  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e5)  # interval to save checkpoint
summary_steps = 2000  # interval to save summary
weight_decay = 1e-5  # weight decay
device = None  # gpu id  (or None)


# other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir_base = './storage'

train_dir = os.path.join(train_dir_base, 'model')  # save directory (Caution!! this folder will be removed at first)

saveparmpath = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters


# =============================================================
# =============================================================

# prepare save folder -----------------------------------------
if train_dir.find('/storage/') > -1:
    if os.path.exists(train_dir):
        print('delete savefolder: %s...' % train_dir)
        shutil.rmtree(train_dir)  # remove folder
    print('make savefolder: %s...' % train_dir)
    os.makedirs(train_dir)  # make folder
else:
    assert False, 'savefolder looks wrong'

# generate sensor signal --------------------------------------
x, _, _, _, _, _, = generate_dataset(num_group=num_group,
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

# train model -------------------------------------------------
if ('data_path' not in locals()) or (data_path is None):
    train(x,
          num_h_nodes=num_h_nodes,
          num_hz_nodes=num_hz_nodes,
          num_hp_nodes=num_hp_nodes,
          phi_type=phi_type,
          phi_share=phi_share,
          initial_learning_rate=initial_learning_rate,
          momentum=momentum,
          max_steps=max_steps,
          decay_steps=decay_steps,
          decay_factor=decay_factor,
          batch_size=batch_size,
          train_dir=train_dir,
          weight_decay=weight_decay,
          checkpoint_steps=checkpoint_steps,
          moving_average_decay=moving_average_decay,
          summary_steps=summary_steps,
          device=device,
          random_seed=random_seed)
else:
    train_3dident(x,  # without observational mixing
                  data_dir=data_path,
                  num_hz_nodes=num_hz_nodes,
                  num_hp_nodes=num_hp_nodes,
                  phi_type=phi_type,
                  initial_learning_rate=initial_learning_rate,
                  momentum=momentum,
                  max_steps=max_steps,
                  decay_steps=decay_steps,
                  decay_factor=decay_factor,
                  batch_size=batch_size,
                  train_dir=train_dir,
                  weight_decay=weight_decay,
                  checkpoint_steps=checkpoint_steps,
                  moving_average_decay=moving_average_decay,
                  summary_steps=summary_steps,
                  device=device,
                  random_seed=random_seed)

# save parameters necessary for evaluation --------------------
model_parm = {'random_seed': random_seed,
              'num_group': num_group,
              'num_dim': num_dim,
              'num_data': num_data,
              'lam1': lam1,
              'lam2': lam2,
              'lamin1': lamin1,
              'lamin2': lamin2,
              'ar_alpha': ar_alpha,
              'ar_beta': ar_beta,
              'num_neighbor': num_neighbor,
              'num_neighbor_in': num_neighbor_in,
              'dag': dag,
              'dist_type': dist_type,
              'num_layer': num_layer,
              'num_h_nodes': num_h_nodes,
              'num_hz_nodes': num_hz_nodes,
              'num_hp_nodes': num_hp_nodes,
              'phi_type': phi_type,
              'phi_share': phi_share,
              'apply_pca': apply_pca,
              'moving_average_decay': moving_average_decay,
              'num_latent': num_latent if 'num_latent' in locals() else None,
              'data_path': data_path if 'data_path' in locals() else None}

print('Save parameters...')
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

# save as tarfile
tarname = train_dir + ".tar.gz"
archive = tarfile.open(tarname, mode="w:gz")
archive.add(train_dir, arcname="./")
archive.close()

print('done.')
