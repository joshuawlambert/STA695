import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import sklearn.preprocessing as pp
import scipy.sparse
import scipy.spatial
from pathlib import Path
import pickle
import awesomeml.dataset as ds

data_dir = Path('D:/data/cora')


def load_cora2(data_dir):
    x = pickle.load(open(data_dir/'ind.cora.x', 'rb'), encoding='latin1')
    y = pickle.load(open(data_dir/'ind.cora.y', 'rb'), encoding='latin1')
    tx = pickle.load(open(data_dir/'ind.cora.tx', 'rb'), encoding='latin1')
    ty = pickle.load(open(data_dir/'ind.cora.ty', 'rb'), encoding='latin1')
    allx = pickle.load(open(data_dir/'ind.cora.allx', 'rb'), encoding='latin1')
    ally = pickle.load(open(data_dir/'ind.cora.ally', 'rb'), encoding='latin1')
    graph = pickle.load(open(data_dir/'ind.cora.graph', 'rb'), encoding='latin1')
    test_idx_reorder = np.loadtxt(data_dir/'ind.cora.test.index', dtype=int)
    test_idx_range = np.sort(test_idx_reorder)
    features = sp.sparse.vstack((allx,tx)).tolil()
    features[test_idx_reorder,:] = features[test_idx_range,:]
    features = np.array(features.todense().astype(np.int))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder,:] = labels[test_idx_range,:]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))

    return dict(features=features,
                categories=labels,
                citation_graph=adj,
                idx_test = idx_test,
                idx_train = idx_train,
                idx_val = idx_val)


def match_graph(graph1, features1, graph2, features2):
    idx2_to_1 = -np.arange(graph1.shape[0])-1
    feature_dist = sp.spatial.distance.cdist(features2, features1)
    feature_dist = feature_dist == 0
    mask_exact_match = np.sum(feature_dist, axis=1) == 1
    idx2_to_1[mask_exact_match] = np.nonzero(feature_dist[mask_exact_match,:])[1]

    for idx2 in np.nonzero(idx2_to_1 < 0)[0]:
        idx1_candi = set(np.nonzero(feature_dist[idx2])[-1])
        idx1_candi -= set(x for x in idx2_to_1 if x >= 0)
        idx1_candi = list(idx1_candi)
        nodes2 = set(idx2_to_1[np.nonzero(graph2[idx2])[-1]])
        score = []
        for idx1 in idx1_candi:
            nodes1 = set(np.nonzero(graph1[idx1])[-1])
            if len(nodes1) == len(nodes2):
                diff_nodes = nodes1 ^ nodes2
                ndiff = len(diff_nodes)
                ndiff -= 2 * sum(x<0 for x in diff_nodes)
                score.append(ndiff/len(nodes1))
            else:
                score.append(np.inf)
        assert min(score)==0

        idx2_to_1[idx2] = idx1_candi[score.index(0)]

    assert np.all(np.sort(idx2_to_1)==np.arange(len(idx2_to_1)))
    features3 = features1[idx2_to_1].copy()
    graph3 = graph1[idx2_to_1,:].copy()
    graph3 = graph3[:,idx2_to_1]
    assert (graph2 != graph3).nnz == 0
    assert np.all(features2 == features3)

    return idx2_to_1

def match_data(data1, data2):
    features1 = data1['features']
    graph1 = data1['citation_graph']
    graph1 += graph1.T
    graph1[graph1>1] = 1

    features2 = data2['features']
    graph2 = data2['citation_graph']
    idx221_mapper = match_graph(graph1, features1, graph2, features2)
    return idx221_mapper

data1 = ds.load_citation(data_dir, 'cora')
data2 = load_cora2(data_dir)
idx221_mapper = match_data(data1, data2)
id_encoder = data1['id_encoder']
id_test = id_encoder.inverse_transform(idx221_mapper[data2['idx_test']])
id_train = id_encoder.inverse_transform(idx221_mapper[data2['idx_train']])
id_val = id_encoder.inverse_transform(idx221_mapper[data2['idx_val']])

ttv_id = list(id_train) + list(id_val) + list(id_test)
ttv_label = ['train']*id_train.shape[0] + ['validation']*id_val.shape[0] + ['test']*id_test.shape[0]

ttv = pd.DataFrame(dict(id=ttv_id, label=ttv_label))
ttv.to_csv(data_dir/"train_validation_test.csv", index=False)
