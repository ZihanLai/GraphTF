import csv
import torch
import random
import numpy as np


def read_mmd_csv(path1, path2):
    all_one_index = []
    m_m_d_true = np.zeros((325, 283, 325))
    with open(path1, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            s = list(map(int, line))
            m_m_d_true[s[0], s[2], s[1]] = 1
            m_m_d_true[s[1], s[2], s[0]] = 1
            all_one_index.append([s[0], s[2], s[1]])
            all_one_index.append([s[1], s[2], s[0]])

    m_m_d_true = torch.FloatTensor(m_m_d_true)

    all_zero_index = []
    with open(path2, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            s = list(map(int, line))
            all_zero_index.append([s[0], s[2], s[1]])

    return m_m_d_true, all_one_index, all_zero_index


def read_sim_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        sim_data = []
        sim_data += [[float(i) for i in row] for row in reader]
    return torch.FloatTensor(sim_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def read_mir_dis(path):
    mir_dis = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            s = list(map(int, line))
            mir_dis.append([s[0], s[1]])

    m_d_tensor = torch.LongTensor(mir_dis)
    return mir_dis, m_d_tensor


def read_mir_mir(path):
    mir_mir = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            s = list(map(int, line))
            mir_mir.append([s[0], s[1]])

    m_m_tensor = torch.LongTensor(mir_mir)
    return mir_mir, m_m_tensor


def read_mmd_one_zero_index(path):
    all_one_index = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            s = list(map(int, line))
            all_one_index.append([s[0], s[2], s[1]])
            all_one_index.append([s[1], s[2], s[0]])

    random.shuffle(all_one_index)

    return all_one_index


def prepare_data(num, i):
    dataset = dict()
    m_m_d_true, all_one_index, all_zero_index = read_mmd_csv('../data/Index_mir_pair_dis_link.csv',
                                                             '../data/negative_sample_{}_{}.csv'.format(num, i))
    dataset['mmd_true'] = m_m_d_true
    dataset['mmd_p'], _, _ = read_mmd_csv('../data/Index_mir_pair_dis_link.csv',
                                          '../data/negative_sample_{}_{}.csv'.format(num, i))
    '''
        if gpu memory is enough
    '''
    # use all positive samples: 15484 items
    random.shuffle(all_one_index)
    random.shuffle(all_zero_index)

    all_one_tensor = torch.LongTensor(all_one_index)
    all_zero_tensor = torch.LongTensor(all_zero_index)

    mm_func = read_sim_csv('../data/325_mir_functional_sim_matrix.csv')
    mm_func_edge_index = get_edge_index(mm_func)
    dataset['mm_func'] = {'data': mm_func, 'edge_index': mm_func_edge_index}

    mm_seq = read_sim_csv('../data/325_mir_seq_sim_matrix.csv')
    mm_seq_edge_index = get_edge_index(mm_seq)
    dataset['mm_seq'] = {'data': mm_seq, 'edge_index': mm_seq_edge_index}

    dd_sem = read_sim_csv('../data/283_dis_semantic_sim_matrix.csv')
    dd_sem_edge_index = get_edge_index(dd_sem)
    dataset['dd_sem'] = {'data': dd_sem, 'edge_index': dd_sem_edge_index}
    print("============================data read over==============================")

    return dataset, all_one_index, all_one_tensor, all_zero_index, all_zero_tensor




