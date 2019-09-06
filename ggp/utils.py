# Author: Yin Cheng Ng
# Some data pipeline and pre-processing functions are derived from https://github.com/tkipf/gcn

import sys, os, zipfile
import scipy.sparse as sp, numpy as np, pickle as pkl, networkx as nx, tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from urllib2 import urlopen, URLError, HTTPError
import GPflow
float_type = GPflow.settings.dtypes.float_type
np_float_type = np.float32 if float_type == tf.float32 else np.float64


def dlfile(url, local_file_path):
    # Open the url
    try:
        f = urlopen(url)
        print "downloading " + url
        # Open our local file for writing
        with open(local_file_path, "wb") as local_file:
            local_file.write(f.read())
    #handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


def check_and_download_dataset(data_name):
    dataset_dir = os.path.join(os.getenv('PWD'), 'Dataset')
    if not(os.path.isdir(dataset_dir)):
        os.mkdir(dataset_dir)
    if data_name == 'citation_networks':
        data_url = 'https://www.dropbox.com/s/tln5wxqqp3o691s/citation_networks.zip?dl=1'
        data_dir = os.path.join(dataset_dir, 'citation_networks')
    else:
        raise RuntimeError('Unsupported dataset {0}'.format(data_name))
    if os.path.isdir(data_dir):
        return True
    else:
        print 'Downloading from '+data_url
        dlfile(data_url, dataset_dir+'/{0}.zip'.format(data_name))
        print 'Download complete. Extracting to '+dataset_dir
        zip_handler = zipfile.ZipFile(dataset_dir+'/{0}.zip'.format(data_name), 'r')
        zip_handler.extractall(dataset_dir)
        zip_handler.close()
        return True


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_data_original(dataset_str, active_learning = False):
    """
    Load data with fixed split as in Planetoid
    :param dataset_str:
    :param active_learning:
    :return:
    """
    data_path = os.getenv('PWD')+'/Dataset/citation_networks/'
    check_and_download_dataset('citation_networks')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_path + "ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if active_learning:
        t = adj.toarray()
        sg = list(nx.connected_component_subgraphs(nx.from_numpy_matrix(t)))
        vid_largest_graph = sg[np.argmax([nx.adjacency_matrix(g).shape[0] for g in sg])].nodes()
        adj = t[vid_largest_graph,:]; adj = adj[:, vid_largest_graph]
        return sp.csr_matrix(adj), sp.csr_matrix(features.toarray()[vid_largest_graph,:]), labels[vid_largest_graph]
    else:
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_base_data(dataset_str):
    data_path = os.getenv('PWD')+'/Dataset/citation_networks/'
    check_and_download_dataset('citation_networks')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_path + "ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    n = labels.shape[0]

    return features, labels, adj, n, idx_train, idx_val, idx_test


def get_training_masks(n, random_split, split_sizes, random_split_seed, add_val, add_val_seed, p_val=0.5, idx_train=None,
                       idx_val=None, idx_test=None):
    if random_split:
        train_mask = []
        val_mask = []
        test_mask = []
    else: # use fixed split as in planetoid
        train_mask = sample_mask(idx_train, n)
        val_mask = sample_mask(idx_val, n)
        test_mask = sample_mask(idx_test, n)
    if add_val:
        train_mask, val_mask = add_val_to_train(train_mask, val_mask, add_val_seed, p_val)

    print("**********************************************************************************************")
    print("train size: {} val size: {} test size: {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))
    print("**********************************************************************************************")

    return train_mask, val_mask, test_mask


def load_data(dataset_str, random_split, split_sizes, random_split_seed, add_val, add_val_seed, p_val, active_learning):
    """
    Load data with fixed split as in Planetoid
    :param dataset_str:
    :param active_learning:
    :return:
    """
    features, labels, adj, n, idx_train, idx_val, idx_test = load_base_data(dataset_str)
    train_mask, val_mask, test_mask = get_training_masks(n, random_split, split_sizes, random_split_seed,
                                                         add_val, add_val_seed, p_val, idx_train, idx_val, idx_test)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if active_learning:
        t = adj.toarray()
        sg = list(nx.connected_component_subgraphs(nx.from_numpy_matrix(t)))
        vid_largest_graph = sg[np.argmax([nx.adjacency_matrix(g).shape[0] for g in sg])].nodes()
        adj = t[vid_largest_graph,:]; adj = adj[:, vid_largest_graph]
        return sp.csr_matrix(adj), sp.csr_matrix(features.toarray()[vid_largest_graph,:]), labels[vid_largest_graph]
    else:
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def add_val_to_train(mask_train, mask_val, seed_val, p=0.5):
    """
    Add a percentage of the validation set to the training set
    :param mask_train:
    :param mask_val:
    :param seed_val:
    :param p: Probability of a point in validation to be addded in training
    :return:
    """
    print("Adding some validation data to training")
    rnd_val = np.random.RandomState(seed_val)
    chs = rnd_val.choice([True, False], size=np.sum(mask_val), p=[p, 1.0 - p])
    mask_val_new = np.array(mask_val)
    mask_train_new = np.array(mask_train)
    mask_val_new[mask_val_new] = chs
    mask_train_new[mask_val] = ~chs
    return mask_train_new, mask_val_new


def load_data_ssl(data_name, random_split=False, split_sizes=None, random_split_seed=1, add_val=False, add_val_seed=1, p_val=0.5):
    adj_csr, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        load_data(dataset_str=data_name, random_split=random_split, split_sizes=split_sizes, random_split_seed=random_split_seed,
                  add_val=add_val, add_val_seed=add_val_seed, p_val=p_val, active_learning=False)

    adj_mat = np.asarray(adj_csr.toarray(), dtype=np_float_type)
    x_tr = np.reshape(np.arange(len(train_mask))[train_mask], (-1, 1))
    x_val = np.reshape(np.arange(len(val_mask))[val_mask], (-1, 1))
    x_test = np.reshape(np.arange(len(test_mask))[test_mask], (-1, 1))
    y_tr = np.asarray(y_train[train_mask], dtype=np.int32)
    y_tr = np.reshape(np.sum(np.tile(np.arange(y_tr.shape[1]), (np.sum(train_mask), 1)) * y_tr, axis=1), (-1, 1))
    y_val = np.asarray(y_val[val_mask], dtype=np.int32)
    y_val = np.reshape(np.sum(np.tile(np.arange(y_val.shape[1]), (np.sum(val_mask), 1)) * y_val, axis=1), (-1, 1))
    y_test = np.asarray(y_test[test_mask], dtype=np.int32)
    y_test = np.reshape(np.sum(np.tile(np.arange(y_test.shape[1]), (np.sum(test_mask), 1)) * y_test, axis=1), (-1, 1))
    node_features = features.toarray()
    if data_name.lower() != 'pubmed': #pubmed already comes with tf-idf
        transformer = TfidfTransformer(smooth_idf=True)
        node_features = transformer.fit_transform(node_features).toarray()
    return adj_mat, node_features, x_tr, y_tr, x_val, y_val, x_test, y_test

def load_data_al(data_name):
    adj_csr, features_csr, labels = load_data(data_name, active_learning=True)
    y = np.sum(np.tile(np.arange(labels.shape[1]), (labels.shape[0], 1)) * labels, axis=1, keepdims=True)
    y = np.asarray(y, dtype=np.int)
    x = np.reshape(np.arange(y.shape[0]), (-1,1))
    adj_mat = np.asarray(adj_csr.toarray(), dtype=np_float_type)
    node_features = features_csr.toarray()
    if data_name.lower() != 'pubmed': #pubmed already comes with tf-idf
        transformer = TfidfTransformer(smooth_idf=True)
        node_features = transformer.fit_transform(node_features).toarray()
    node_features = node_features - np.mean(node_features, axis=0)
    return adj_mat, node_features, x, y