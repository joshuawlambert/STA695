import time
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as pp
from pathlib import Path
import awesomeml as aml
import sys

# ************************************************************
# global settings
# ************************************************************
ckpt_file = Path('ckpt/gat/gat_cora.ckpt')
data_name = 'cora'
data_dir = Path('~/data/cora')

nhop = 1
if len(sys.argv) > 1:
    nhop = int(sys.argv[1])
acc_fname = 'acc_gat2_nhop_{}.txt'.format(nhop)
if len(sys.argv) > 2:
    acc_fname = str(sys.argv[2])

# training params
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

adj_eye = np.asarray(adj + np.eye(N))
adj_T_eye = np.asarray(adj.T + np.eye(N))
adj_all_eye = np.asarray(adj + adj.T + np.eye(N))
tmp = np.dstack((adj_eye, adj_T_eye))
for i in range(nhop-1):
    tmp = np.dstack((tmp, np.linalg.matrix_power(adj_eye, i+2)))
    tmp = np.dstack((tmp, np.linalg.matrix_power(adj_T_eye, i+2)))
    tmp = np.dstack((tmp, np.linalg.matrix_power(adj_all_eye, i+2)))
    
bias = np.full_like(tmp, -1e9)
bias[tmp>0] = 0

mask_train = np.zeros(N, dtype=np.bool)
mask_train[data['id_encoder'].transform(tvt['id'][tvt['label']=='train'])]=1
mask_val = np.zeros(N, dtype=np.bool)
mask_val[data['id_encoder'].transform(tvt['id'][tvt['label']=='validation'])]=1
mask_test = np.zeros(N, dtype=np.bool)
mask_test[data['id_encoder'].transform(tvt['id'][tvt['label']=='test'])]=1

nb_nodes = N
ft_size = P
nb_classes = NC

# ************************************************************
# construct computing graph
# ************************************************************
X = X[np.newaxis]
Y = Y[np.newaxis]
bias = bias[np.newaxis]
mask_train = mask_train[np.newaxis]
mask_val = mask_val[np.newaxis]
mask_test = mask_test[np.newaxis]

tf.reset_default_graph()
with tf.name_scope('input'):
    ftr_in = tf.placeholder(dtype=tf.float32, shape=(1,N,P))
    bias_in = tf.placeholder(dtype=tf.float32, shape=bias.shape)
    lbl_in = tf.placeholder(dtype=tf.int32, shape=(1,N,NC))
    msk_in = tf.placeholder(dtype=tf.int32, shape=(1,N))
    training = tf.placeholder(dtype=tf.bool, shape=())

attns = []
for _ in range(n_heads[0]):
    attns.append(aml.layers.gat2(
        ftr_in, bias_mat=bias_in, out_sz=hid_units[0],
        activation=nonlinearity, in_drop=fea_drop, coef_drop=coef_drop,
        residual=False, training=training))
    h_1 = tf.concat(attns, axis=-1)

for i in range(1, len(hid_units)):
    h_old = h_1
    attns = []
    for _ in range(n_heads[i]):
        attns.append(aml.layers.gat2(
            h_1, bias_mat=bias_in, out_sz=hid_units[i],
            activation=nonlinearity, in_drop=fea_drop,
            coef_drop=coef_drop, residual=residual, training=training))
    h_1 = tf.concat(attns, axis=-1)
out = []
for i in range(n_heads[-1]):
    out.append(aml.layers.gat2(
        h_1, bias_mat=bias_in, out_sz=NC, activation=lambda x: x,
        in_drop=fea_drop, coef_drop=coef_drop, residual=False, training=training, reduce=True))
logits = tf.add_n(out) / n_heads[-1]

log_resh = tf.reshape(logits, [-1, NC])
lab_resh = tf.reshape(lbl_in, [-1, NC])
msk_resh = tf.reshape(msk_in, [-1])
msk_resh = tf.cast(msk_resh, dtype=tf.float32)
msk_resh /= tf.reduce_mean(msk_resh)
if hasattr(tf.nn, 'softmax_cross_entropy_with_logits_v2'):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=log_resh, labels=lab_resh)
else:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=log_resh, labels=lab_resh)
loss *= msk_resh
loss = tf.reduce_mean(loss)

tmp = tf.equal(tf.argmax(log_resh,1), tf.argmax(lab_resh,1))
tmp = tf.cast(tmp, tf.float32)
tmp *= msk_resh
accuracy = tf.reduce_mean(tmp)

vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not in
                   ['bias', 'gamma', 'b', 'g', 'beta']]) *l2_coef
opt = tf.train.AdamOptimizer(learning_rate=lr)
train_op = opt.minimize(loss+lossL2)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# ************************************************************
# training
# ************************************************************
vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0

saver = tf.train.Saver()
ckpt_file.parent.mkdir(parents=True, exist_ok=True)
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(nb_epochs):
        _, loss_tr, acc_tr = sess.run(
            [train_op, loss, accuracy],
            feed_dict={
                ftr_in: X,
                bias_in: bias,
                lbl_in: Y,
                msk_in: mask_train,
                training: True})

        loss_vl, acc_vl = sess.run(
            [loss, accuracy],
            feed_dict={
                ftr_in: X,
                bias_in: bias,
                lbl_in: Y,
                msk_in: mask_val,
                training: False})
        print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' % (loss_tr, acc_tr, loss_vl, acc_vl))

        if acc_vl >= vacc_mx or loss_vl <= vlss_mn:
            if acc_vl >= vacc_mx and loss_vl <= vlss_mn:
                vacc_early_model = acc_vl
                vlss_early_model = loss_vl
                saver.save(sess, str(ckpt_file))
            vacc_mx = np.max((acc_vl, vacc_mx))
            vlss_mn = np.min((loss_vl, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                break

    saver.restore(sess, str(ckpt_file))

    loss_ts, acc_ts = sess.run(
        [loss, accuracy],
        feed_dict={
            ftr_in: X,
            bias_in: bias,
            lbl_in: Y,
            msk_in: mask_test,
            training: False})

    with open(acc_fname, 'a') as file:
        file.write('{} '.format(acc_ts))
        
    print('Test loss:', loss_ts, '; Test accuracy:', acc_ts)
    sess.close()
