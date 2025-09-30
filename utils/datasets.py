import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import json
from PIL import Image
import time
import numpy as np
import random
from random import randint
import ipdb

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
import unicodedata


def unicode_escaping(s):
    result = ""
    for c in s:
        if unicodedata.category(c) in ("Mn", "Mc"):
            result += f"#U{ord(c):04x}"
        else:
            result += c
    return result


class artEmisXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len, len_prefix=0,use_explanation=True,prompt_text=''):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # prefix + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.len_prefix = len_prefix
        self.n_emotions=9
        self.prompt_text=prompt_text 
        self.index_tracker = {k: len(v['explanations']) - 1 for k,v in self.data.items()}
        

    def __getitem__(self, i):
        ta=time.time()
        data_id = self.ids_list[i]
        sample = self.data[data_id]
        img_name = sample['image_name']+'.jpg' 

        exp_idx = self.index_tracker[data_id]  
        if exp_idx > 0:
            self.index_tracker[data_id] -= 1    
        if exp_idx == 0:
            self.index_tracker[data_id] = len(sample['explanations']) - 1

        emotion_1 = sample['emotions'][exp_idx]
        text_exp_1 = sample['explanations'][exp_idx]   

        
        emotion_idx_1 = EMOTION_TO_IDX[emotion_1]  # 将情感映射到索引
        gt_emotion_distribution_1 = sample['image_emotion_distribution'][emotion_idx_1]


        neg_sample={'emotions':[],'explanations':[], 'image_emotion_distribution':[]}
        for e,text in zip(sample['emotions'],sample['explanations']):
            if e not in emotion_1:
                emotion_idx = EMOTION_TO_IDX[e]  # Get the index for the current emotion
                distrib_value = sample['image_emotion_distribution'][emotion_idx]  # Get the corresponding distribution value
                neg_sample['emotions'].append(e)
                neg_sample['explanations'].append(text)
                neg_sample['image_emotion_distribution'].append(distrib_value)
        if not neg_sample['emotions']:
            print("No valid negative samples found. Skipping...")
        else:
            index_num=randint(0,len(neg_sample['emotions'])-1)
            emotion_2=neg_sample['emotions'][index_num]
            text_exp_2=neg_sample['explanations'][index_num]
            gt_emotion_distribution_2 = neg_sample['image_emotion_distribution'][index_num]    
            

        emo_labels_1 = torch.tensor([emotion_idx_1], dtype=torch.long)
        emotion_idx_2 = EMOTION_TO_IDX[emotion_2]
        emo_labels_2 = torch.tensor([emotion_idx_2], dtype=torch.long)
        emo_labels = torch.stack([emo_labels_1,emo_labels_2])


        # tokenization process
        pf_segment_id, emo_segment_id, exp_segment_id = self.tokenizer.convert_tokens_to_ids(['<prefix>', 
                                                                                              '<emotion>', 
                                                                                         '<explanation>'])

        pf_token =  '<prefix>'  # prefix is the learable embeddings produced by a prefix mappting net.                                                          
        tokens= [pf_token] * self.len_prefix                                                                                                                                      
        labels = [-100] * self.len_prefix   # we dont want to predict the prefix, set to pad to ignore in XE
        segment_ids = [pf_segment_id] * self.len_prefix
        
        
        emotion_1 = [self.tokenizer.bos_token] + self.tokenizer.tokenize(self.prompt_text + " " + emotion_1)
        # emotion_1 = self.tokenizer.tokenize(self.prompt_text + " " + emotion_1)
        emotion_len_1 = len(emotion_1)
        tokens_1 = tokens+emotion_1 

        emotion_2 = [self.tokenizer.bos_token] + self.tokenizer.tokenize(self.prompt_text + " " + emotion_2)
        emotion_len_2 = len(emotion_2)
        tokens_2 = tokens+emotion_2 

 
        tokens_exp_1 = self.tokenizer.tokenize(" because " + text_exp_1) + [self.tokenizer.eos_token]
        exp_len_1 = len(tokens_exp_1)
        tokens_1 += tokens_exp_1
        labels_1 = labels+ [-100] + emotion_1[1:] + tokens_exp_1   # labels will be shifted in the model, so for now set them same as tokens
        segment_ids_1 = segment_ids + [emo_segment_id] * emotion_len_1
        segment_ids_1 = segment_ids_1 + [exp_segment_id] * exp_len_1

        tokens_exp_2 = self.tokenizer.tokenize(" because " + text_exp_2) + [self.tokenizer.eos_token]
        exp_len_2 = len(tokens_exp_2)
        tokens_2 += tokens_exp_2
        labels_2 = labels+ [-100] + emotion_2[1:] + tokens_exp_2   # labels will be shifted in the model, so for now set them same as tokens
        segment_ids_2 = segment_ids + [emo_segment_id] * emotion_len_2
        segment_ids_2 = segment_ids_2 + [exp_segment_id] * exp_len_2

        if len(tokens_1) > self.max_seq_len :
            tokens_1 = tokens_1[:self.max_seq_len]
            labels_1 = labels_1[:self.max_seq_len]
            segment_ids_1 = segment_ids_1[:self.max_seq_len]

        if len(tokens_2) > self.max_seq_len :
            tokens_2 = tokens_2[:self.max_seq_len]
            labels_2 = labels_2[:self.max_seq_len]
            segment_ids_2 = segment_ids_2[:self.max_seq_len]

        assert len(tokens_1) == len(segment_ids_1) 
        assert len(tokens_1) == len(labels_1)
        assert len(tokens_2) == len(segment_ids_2) 
        assert len(tokens_2) == len(labels_2)
        
        seq_len_1 = len(tokens_1)
        padding_len_1 = self.max_seq_len - seq_len_1
        tokens_1 = tokens_1 + ([self.tokenizer.pad_token] * padding_len_1)
        labels_1 = labels_1 + ([-100] * padding_len_1) 
        segment_ids_1 += ([exp_segment_id] * padding_len_1)
        input_ids_1 = self.tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long)
        labels_1 = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels_1]
        labels_1 = torch.tensor(labels_1, dtype=torch.long)
        segment_ids_1 = torch.tensor(segment_ids_1, dtype=torch.long)
    
        seq_len_2 = len(tokens_2)
        padding_len_2 = self.max_seq_len - seq_len_2
        tokens_2 = tokens_2 + ([self.tokenizer.pad_token] * padding_len_2)
        labels_2 = labels_2 + ([-100] * padding_len_2) 
        segment_ids_2 += ([exp_segment_id] * padding_len_2)
        input_ids_2 = self.tokenizer.convert_tokens_to_ids(tokens_2)
        input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long)
        labels_2 = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels_2]
        labels_2 = torch.tensor(labels_2, dtype=torch.long)
        segment_ids_2 = torch.tensor(segment_ids_2, dtype=torch.long)

        folder = 'wikiart_rescaled_max_size_to_600px_same_aspect_ratio/'

        img_path = folder + img_name
        try:
            img = Image.open(img_path)
        except:
            nfc= unicodedata.normalize('NFC', img_path)
            img = Image.open(nfc)
      
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        did = torch.LongTensor([int(data_id)])
        img = torch.stack([img,img])

        input_ids=torch.stack([input_ids_1, input_ids_2])
        labels = torch.stack([labels_1,labels_2])
        segment_ids = torch.stack([segment_ids_1,segment_ids_2])
        gt_emotion_distribution = torch.tensor([gt_emotion_distribution_1,gt_emotion_distribution_2],dtype=torch.float)

        return (img, did, input_ids, labels, segment_ids,None,gt_emotion_distribution, emo_labels)

    def __len__(self):
        return len(self.ids_list)

 
class artEmisXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len,len_prefix=0,\
                 prompt_text=''):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # prefix + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.len_prefix=len_prefix
        self.n_emotions=9
        self.prompt_text=prompt_text


    def __getitem__(self, i):
        
        data_id = self.ids_list[i]
        sample = self.data[data_id]
        img_name = sample['image_name']+'.jpg'  

        # tokenization process
        pf_segment_id, emo_segment_id, exp_segment_id = self.tokenizer.convert_tokens_to_ids(['<prefix>', '<emotion>', '<explanation>'])    
        pf_token =  '<prefix>'  # prefix is the learable embeddings produced by a prefix mappting net.                                                          
        tokens= [pf_token] * self.len_prefix                                                                                                                                      
        segment_ids = [pf_segment_id] * self.len_prefix

        
        gt_emotion_one_hot = np.zeros(self.n_emotions, dtype=np.float32)
        gt_emotion = max(set(sample['emotions']),key=sample['emotions'].count)
        gt_emotion_one_hot[EMOTION_TO_IDX[gt_emotion]] = 1  # [0,0,...,1,...,0]   
        gt_emotion_one_hot = torch.tensor(gt_emotion_one_hot, dtype=torch.float)
        
        
        gt_emotion_distribution = sample['image_emotion_distribution']
        gt_emotion_distribution = torch.tensor(gt_emotion_distribution, dtype=torch.float)
        
        image_emotion_distribution =sample['image_emotion_distribution']
        image_emotion_distribution = torch.tensor(image_emotion_distribution,dtype=torch.float)

        emotion = [self.tokenizer.bos_token] + self.tokenizer.tokenize(self.prompt_text)
        emotion_len = len(emotion )
        tokens += emotion 

        segment_ids += [emo_segment_id] * emotion_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
       
        
        folder = 'wikiart_rescaled_max_size_to_600px_same_aspect_ratio/'
        img_path = folder + img_name

        try:
            img = Image.open(img_path)
        except:
            nfc= unicodedata.normalize('NFC', img_path)
            img = Image.open(nfc)

        if img.mode is not 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        did = torch.LongTensor([int(data_id)])
        
        return (img, did, input_ids, segment_ids, None)


    def __len__(self):
        return len(self.ids_list)



