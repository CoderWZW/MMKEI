
import random
from random import shuffle,randint,choice,sample
import numpy as np

import torch

def next_batch_pairwise(data,batch_size,n_negs=1):

    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y


def next_batch_sequence(data, batch_size,labelgap,max_len=50 ):
    training_data = [item[1] for item in data.original_seq]
    print(len(training_data))
    #源代码
    # shuffle(training_data)

    #改动后
    shuffled_indices = torch.randperm(len(training_data))  # 获取打乱后的索引顺序
    # training_data = training_data[shuffled_indices]  # 对 tensor1 执行 shuffle 操作
    # 将相同的顺序应用到

    labelgap = [ labelgap[i] for i in shuffled_indices]
    training_data= [training_data[i] for i in shuffled_indices]

    ptr = 0
    data_size = len(training_data)
    labelgap=torch.tensor(labelgap)

    item_list = list(range(1,data.item_num+1))

    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seqfull=np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        labelgap_batch=np.zeros((batch_end-ptr, max_len),dtype=np.int)
        posfull=np.zeros((batch_end-ptr, max_len),dtype=np.int)
        y =np.zeros((batch_end-ptr, max_len),dtype=np.int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seqfull[n, :end+1]=training_data[ptr + n][start:]

            labelgap_batch[n,:]=labelgap[ptr+n]


            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            posfull[n, :end+1]= list(range(1,end+2))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end

        yield seq, seqfull,pos,posfull, y, neg, np.array(seq_len,np.int),labelgap_batch


def next_batch_sequence_for_test(data, batch_size,label_gap,max_len=50):
    labelgap = torch.tensor(label_gap)
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        labelgap_batch = np.zeros((batch_end - ptr, max_len), dtype=np.int)
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []

        for n in range(0, batch_end-ptr):

            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end =  len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            labelgap_batch[n, :] = labelgap[ptr + n]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))

        ptr=batch_end
        #没有打乱
        # print(seq[0,:10])
        yield seq, pos, np.array(seq_len,np.int),labelgap_batch