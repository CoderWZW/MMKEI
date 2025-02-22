import csv
import re
import pandas as pd
import numpy as np
from re import split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import json
# from llam_text_to_tensor.main  import  preprocessing_for_llama
import time

max_len = 100


def text_preprocessing_for_llama(s):
    s = s.lower()

    s = re.sub(r"\'t", " not", s)
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    s = re.sub(r'\s+', ' ', s).strip()
    s="Extract useful information for recommendations from the textual descriptions of item," \
      " compress it into a single word, and store it in <EMB>. " \
      "The item description is:"+s
    return s

def text_preprocessing(s):
    s = s.lower()

    s = re.sub(r"\'t", " not", s)
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def preprocessing_for_bert(data):
    from transformers import BertTokenizer, GPT2Tokenizer
    from transformers import BertModel, GPT2LMHeadModel
    from transformers import AutoTokenizer, AutoModel

    # tokenizer = BertTokenizer.from_pretrained('bert', do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('all-mpnet-base-v2', do_lower_case=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),
            add_special_tokens=True,
            max_length=max_len ,
            pad_to_max_length=True,
            # return_tensors='pt',
            return_attention_mask=True ,
        )
        # Add the outputs to the lists
        
        # if(len(input_ids)==0):
        #     input_ids.append(encoded_sent.get('input_ids'))
        # if(len(attention_masks)==0):
        #     attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # [total sentence, 100]
    return input_ids, attention_masks


#下面的函数根据config里数据集名称自动处理
#在这里加载特征数据集，把序列里的物品ID号和特征对应起来，在这里只取了新闻标题，可以取更多
def MINDsmall(id2item):
    featureOriginal = {}
    featureNew=[]
    with open(r"./dataset/MINDsmall/news.tsv", 'r', encoding="gbk",
              errors="ignore") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            #row[0]是原始的未加-的新闻号
            featureOriginal[row[0]] = row[1] + ' ' + row[2] + row[3]
    for i in id2item:
            featureNew.append(featureOriginal[split('-',id2item[i])[0]])
            # print(featureOriginal[split('-',id2item[i])[0]])
            # print(featureNew)
    train_inputs, train_masks = preprocessing_for_bert(featureNew)
    return train_inputs,train_masks




def Amazon():
    featureOriginal = {}

    path='/usr/gao/cwh/ModalRec/ModalRec/dataset/Amazon-Music/'
    csv.field_size_limit(10000000)
    with open(path+"meta_Musical_Instruments.json", 'r', encoding="gbk",
              errors="ignore") as tsv_file:

        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        #for row in tsv_file.readlines():

        for row in tsv_reader:
            y=1
            try:
                title = re.findall('"title": (.*?), "also_buy"', row[0])[0]
                # description = re.findall('"description": [(.*?)], "fit"', row[0])[0]
                # print(description)
                asin=re.findall('"asin": "(.*?)",', row[0])[0]

                # print(asin)
                featureOriginal[asin]=title
            except:
                y+=1
                print(y)
        print(len(featureOriginal))
        json_str = json.dumps(featureOriginal)
        with open(path+'featuretitle.json', 'w') as f:
            f.write(json_str)
            # if('asin' in row.keys()):
            #     featureOriginal[row["asin"]] =' '.join(row["description"])
            # elif('details' in row.keys()):
            #     featureOriginal[row["details"]["asin"]] = ' '.join(row["description"])



def load_feature(id2item,datasetfile):
    with open(datasetfile+'featureNew.json') as f:
        json_str = f.read()
    featureNew=[]
    featureOriginal  = json.loads(json_str)
    t = 0
    for i in id2item:

        try:
            featureNew.append(featureOriginal[id2item[i]])
        except:
            t += 1
            print(t)
            featureNew.append('None')
        # print(featureOriginal[split('-',id2item[i])[0]])
        # print(featureNew)
    # train_inputs, train_masks = preprocessing_for_bert(featureNew)
    train_inputs, train_masks,emb_id= preprocessing_for_llama(featureNew)

    return train_inputs, train_masks,emb_id

if __name__ == "__main__":
     Amazon()
     # load_feature()