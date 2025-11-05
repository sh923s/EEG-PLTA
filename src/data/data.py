import os
from sklearn import preprocessing
import numpy as np
from torch.utils.data import TensorDataset
import torch
from .Dataset_Idx import Dataset_Idx
from scipy.linalg import fractional_matrix_power
import pandas as pd
import scipy.io as sio
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann

def riemann_align(X_train, X_test):
    cov_train = Covariances().fit_transform(X_train[:, 0, :, :])
    cov_test = Covariances().fit_transform(X_test[:, 0, :, :])
    cov_train_aligned = np.zeros(cov_train.shape)
    cov_test_aligned = np.zeros(cov_test.shape)
    num_train_subjects = X_train.shape[0] // (X_test.shape[0])
    trial_per_subject = X_test.shape[0]
    for i in range(num_train_subjects):
        cov_train_sub = cov_train[i * trial_per_subject:(i + 1) * trial_per_subject]
        mean_cov_sub = mean_riemann(cov_train_sub)
        sqrt_mean_cov_sub = fractional_matrix_power(mean_cov_sub, -0.5)
        for j in range(trial_per_subject):
            cov_train_aligned[i * trial_per_subject + j] = sqrt_mean_cov_sub @ cov_train_sub[j] @  sqrt_mean_cov_sub
    # align test data
    cov_test_sub = cov_test
    mean_cov_sub = mean_riemann(cov_test_sub)
    sqrt_mean_cov_sub = fractional_matrix_power(mean_cov_sub, -0.5)
    for j in range(cov_test.shape[0]):
        cov_test_aligned[j] = sqrt_mean_cov_sub @ cov_test_sub[j] @  sqrt_mean_cov_sub
    return cov_train_aligned, cov_test_aligned
        
def coral(x_src, x_tar):
    """
    Parameters
    ----------
    x_src : numpy array
        source data of shape (num_samples, num_channels, num_time_samples)
    x_tar : numpy array
        target data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    Xsrc_coral : numpy array
        aligned source data of shape (num_samples, num_channels, num_time_samples)
    """
    cov_src = np.zeros((x_src.shape[0], x_src.shape[1], x_src.shape[1]))
    cov_tar = np.zeros((x_tar.shape[0], x_tar.shape[1], x_tar.shape[1]))
    for i in range(x_src.shape[0]):
        cov_src[i] = np.cov(x_src[i])
    for i in range(x_tar.shape[0]):
        cov_tar[i] = np.cov(x_tar[i])
    refSrc = np.mean(cov_src, 0)
    refTar = np.mean(cov_tar, 0)
    sqrtRefSrc = fractional_matrix_power(refSrc, -0.5)
    sqrtRefTar = fractional_matrix_power(refTar, 0.5)
    Xsrc_coral = np.zeros(x_src.shape)
    for i in range(x_src.shape[0]):
        Xsrc_coral[i] = np.dot(sqrtRefTar, np.dot(sqrtRefSrc, x_src[i]))
    return Xsrc_coral
    

def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def preprocess_data(X, y, num_subject, target_subject_id, align):
    align_out = []
    subject_trial = X.shape[0] // num_subject
    # process data
    if align == 'ea':
        for i in range(num_subject):
            X_sub = X[i * subject_trial:(i + 1) * subject_trial]
            X_sub_aligned = EA(X_sub) # euclidean alignment
            align_out.append(X_sub_aligned)
        X = np.concatenate(align_out, axis=0)
    elif align == 'coral':
        # align each subject to the target subject
        X_target = X[target_subject_id * subject_trial:(target_subject_id + 1) * subject_trial]
        for i in range(num_subject):
            X_sub = X[i * subject_trial:(i + 1) * subject_trial]
            X_sub_aligned = coral(X_sub, X_target) # coral alignment
            align_out.append(X_sub_aligned)
        X = np.concatenate(align_out, axis=0)
    X = np.expand_dims(X, axis=1)
    # process label
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y).reshape(-1)
    
    return X, y

def traintest_split_cross_subject(X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    return train_x, train_y, test_x, test_y

def load_dataset(args, session, align):
    if args.dset == 'BNCI2014001':
        X, y = load_BNCI2014001(args.dset_root, args.subject_num, args.target_subject_id, session, align)
    elif args.dset == 'BNCI2014002':
        X, y = load_BNCI2014002(args.dset_root, args.subject_num, args.target_subject_id, session, align)
    elif args.dset == 'BNCI2015001':
        X, y = load_BNCI2015001(args.dset_root, args.subject_num, args.target_subject_id, session, align)
    elif args.dset == 'BNCI2014001-4':
        X, y = load_BNCI2014001_4(args.dset_root, args.subject_num, args.target_subject_id, session, align)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dset))
    # generate adversarial sample in the dataset
    if args.scenario == 'noise':
        X = add_noise(X, args.snr, args.noise_channel_ratio)
    if args.scenario == 'channel_shuffle':
        X = shuffle_channels(X, args.shuffle_channel_ratio)
    if args.scenario == 'amplitude_scaling':
        X = amplitude_scaling(X, args.maximum_amplitude_scaling)
    # split data
    train_x, train_y, test_x, test_y = traintest_split_cross_subject(X, y, args.subject_num, args.target_subject_id)
    # convert to tensor
    train_x = torch.from_numpy(train_x).to(torch.float32)
    train_y = torch.from_numpy(train_y).to(torch.long)
    test_x = torch.from_numpy(test_x).to(torch.float32)
    test_y = torch.from_numpy(test_y).to(torch.long)
    # create dataset
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    return train_dataset, test_dataset

def load_dataset_idx(args, pre_align):
    train_dataset, test_dataset = load_dataset(args, pre_align)
    train_dataset_idx = Dataset_Idx(train_dataset)
    test_dataset_idx = Dataset_Idx(test_dataset)
    return train_dataset_idx, test_dataset_idx

def load_BNCI2014001(root, num_subject, target_subject_id, session, align):
    X = np.load(os.path.join(root, 'BNCI2014001', 'X.npy'))
    y = np.load(os.path.join(root, 'BNCI2014001', 'labels.npy'))
    indices = []
    for i in range(num_subject):
        if session == 1:
            indices.append(np.arange(288) + (576 * i))
        elif session == 2:
            indices.append(np.arange(288, 576) + (576 * i))
    indices = np.concatenate(indices, axis=0)
    X = X[indices]
    y = y[indices]
    indices = []
    for i in range(len(y)):
        if y[i] in ['left_hand', 'right_hand']:
            indices.append(i)
    X = X[indices]
    y = y[indices]
    
    return preprocess_data(X, y, num_subject, target_subject_id, align)

def load_BNCI2014001_4(root, num_subject, target_subject_id, session, align):
    X = np.load(os.path.join(root, 'BNCI2014001-4', 'X.npy'))
    y = np.load(os.path.join(root, 'BNCI2014001-4', 'labels.npy'))
    indices = []
    for i in range(num_subject):
        if session == 1:
            indices.append(np.arange(288) + (576 * i))
        elif session == 2:
            indices.append(np.arange(288, 576) + (576 * i))
    indices = np.concatenate(indices, axis=0)
    X = X[indices]
    y = y[indices]
    
    return preprocess_data(X, y, num_subject, target_subject_id, align)

def load_BNCI2014002(root, num_subject, target_subject_id, session, align):
    X = np.load(os.path.join(root, 'BNCI2014002', 'X.npy'))
    y = np.load(os.path.join(root, 'BNCI2014002', 'labels.npy'))
    indices = []
    for i in range(num_subject):
        if session == 1:
            indices.append(np.arange(100) + (160 * i))
        elif session == 2:
            indices.append(np.arange(100, 160) + (160 * i))
    indices = np.concatenate(indices, axis=0)
    X = X[indices]
    y = y[indices]
    
    return preprocess_data(X, y, num_subject, target_subject_id, align)

def load_BNCI2015001(root, num_subject, target_subject_id, session, align):
    X = np.load(os.path.join(root, 'BNCI2015001', 'X.npy'))
    y = np.load(os.path.join(root, 'BNCI2015001', 'labels.npy'))
    indices = []
    for i in range(num_subject):
        if i in [7, 8, 9, 10, 11]:
            if session == 1:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif session == 2:
                indices.append(np.arange(200, 400) + (400 * 7) + 600 * (i - 7))
        else:
            if session == 1:
                indices.append(np.arange(200) + (400 * i))
            elif session == 2:
                indices.append(np.arange(200, 400) + (400 * i))

    indices = np.concatenate(indices, axis=0)
    X = X[indices]
    y = y[indices]
    
    return preprocess_data(X, y, num_subject, target_subject_id, align)

def wgn(x, snr):
    power = np.mean(x ** 2)
    noise_power = power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), x.shape)
    return x + noise

def add_noise(X, snr, noise_channel_ratio):
    noise_channels = np.random.choice(X.shape[2], int(X.shape[2] * noise_channel_ratio), replace=False)
    for trial_id in range(X.shape[0]):
        for ch_id in noise_channels:
            X[trial_id, 0, ch_id, :] = wgn(X[trial_id, 0, ch_id, :], snr)
    return X

def shuffle_channels(X, shuffle_ratio):
    shuffle_channels = np.random.choice(X.shape[2], int(X.shape[2] * shuffle_ratio), replace=False)
    permuted_channels = np.random.permutation(shuffle_channels)
    for trial_id in range(X.shape[0]):
        X[trial_id, 0, shuffle_channels, :] = X[trial_id, 0, permuted_channels, :]
    return X

def amplitude_scaling(X, max_scaling):
    scaling_factors = np.random.uniform(0, max_scaling, X.shape[2])
    for trial_id in range(X.shape[0]):
        for ch_id in range(X.shape[2]):
            X[trial_id, 0, ch_id, :] = X[trial_id, 0, ch_id, :] * scaling_factors[ch_id]
    return X