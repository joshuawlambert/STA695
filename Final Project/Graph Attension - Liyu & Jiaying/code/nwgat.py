import time
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as pp
from pathlib import Path
import awesomeml as aml

# ************************************************************
# global settings
# ************************************************************
ckpt_file = Path('ckpt/gat/gat_cora.ckpt')
data_name = 'cora'
data_dir = Path('~/data/cora')

# training params
batch_size = 1
nb_epochs = 100000
#nb_epochs = 2
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
fea_drop = 0.6
coef_drop = 0.6

print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))

# ************************************************************
# prepare data
# ************************************************************
data = aml.dataset.load_citation(data_dir, 'cora')
tvt = aml.dataset.load_citation_tvt('cora')

X = data['features']
X = X / X.sum(axis=1).reshape((-1,1))
Y = pp.label_binarize(data['categories'], classes=list(set(data['categories'])))
N = X.shape[0]
P = X.shape[1]
NC = Y.shape[1]
assert N == Y.shape[0]

adj = data['citation_graph'].todense()
bias = np.full_like(adj, -1e9)
bias[np.logical_or(adj+np.eye(N) > 0, adj.T+np.eye(N) > 0)] = 0

mask_train = np.zeros(N, dtype=np.bool)
mask_train[data['id_encoder'].transform(tvt['id'][tvt['label']=='train'])]=1
mask_val = np.zeros(N, dtype=np.bool)
mask_val[data['id_encoder'].transform(tvt['id'][tvt['label']=='validation'])]=1
mask_test = np.zeros(N, dtype=np.bool)
mask_test[data['id_encoder'].transform(tvt['id'][tvt['label']=='test'])]=1

nb_nodes = N
ft_size = P
nb_classes = NC

X = X[np.newaxis]
Y = Y[np.newaxis]
bias = bias[np.newaxis]
mask_train = mask_train[np.newaxis]
mask_val = mask_val[np.newaxis]
mask_test = mask_test[np.newaxis]

# ************************************************************
# construct computing graph
# ************************************************************
def nwgat(node, neighbor, ntrans, activation=None, in_drop=0.0, coef_drop=0.0, training=False):
    nchannels = node.get_shape()[-1]
    assert nchannels == neighbor.get_shape()[-1]

    if in_drop != 0.0:
        node = tf.layers.dropout(node, in_drop, training=training)
        neighbor = tf.layers.dropout(neighbor, coef_drop, training=training)

    node = tf.expand_dims(node, axis=1) # Nx1xF
    W = tf.get_variable('W', shape=(nchannels, ntrans)) # FxF'
    node = tf.matmul(node, W) # Nx1xF'
    neighbor = tf.matmul(neighbor, W) # NxKxF'

    node = tf.layers.conv1d(node, 1, 1) # Nx1x1
    neighbor = tf.layers.conv1d(neighbor, 1, 1) # NxKx1
    logits = node + neighbor # NxKx1
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=-2) # NxKx1
    coefs = tf.transpose(coefs, [0, 2, 1]) # Nx1xK

    if coef_drop != 0.0:
        coefs = tf.layers.dropout(coefs, coef_drop, training=training)
    if in_drop != 0.0:
        node = tf.layers.dropout(node, in_drop, training=training)
        neighbor = tf.layers.dropout(neighbor, coef_drop, training=training)

    vals = tf.squeeze(tf.matmul(coefs, neighbor), axis=1) # NxF'
    
    


    
