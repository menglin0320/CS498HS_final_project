from project_configuration import config
import json
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os

def basic_sentence_split(words):
    sentences = []
    for i in range(0, len(words), 10):
        sentences.append(words[i:i+10])
    return sentences

def sentence2vector(sentences, word_dict, freqency_dict):
    vectors = []
    a = 0.00001
    for s in sentences:
        tmp = np.zeros(300)
        for word in s:
            tmp += a/(a + freqency_dict[word]) * word_dict[word]
        tmp/=len(s)
        vectors.append(tmp.tolist())
    return vectors

def get_dummy_dict(word_dict):
    dummy_dict = {}
    for w in word_dict:
        dummy_dict[w] = 1
    return dummy_dict

def words2vecs(in_array, word_dict, freqency_dict):
    output_list = in_array.tolist()
    for i in range(len(in_array)):
        words = in_array[i][1].split(' ')
        sentences = basic_sentence_split(words)
        sentence_vectors = sentence2vector(sentences, word_dict, freqency_dict)
        output_list[i][1] = sentence_vectors
    return output_list

def get_word2vec_dict(word_embedding_path):
    word_embedding_file = open(word_embedding_path, 'r')
    word_dict = json.load(word_embedding_file)
    for w in word_dict:
        word_dict[w] = [float(num_str) for num_str in word_dict[w].split('')]
    return word_dict

def get_freqency_dict(word_dict, frequency_path = ""):
    if freqency_path=="":
        return get_dummy_dict(word_dict)

    freqency_file = open(freqency_path, 'r')
    freqency_dict = {}
    line = freqency_file.readline()
    while line!='':
        line = line.split(' ')
        if(line[0] in word_dict):
            freqency_dict[line[0]] = float(line[1])
        line = freqency_file.readline()
    return freqency_dict

def split_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current Dir:"+current_dir)
    sample_path = current_dir + 'data/processed_data/processed_attributes.txt'
    label_path = current_dir + 'data/processed_data/labels.txt'
    
    with open(sample_path, 'r') as f1:
        lines=f1.readlines()
    data_list=[]
    for line in lines:
        tempo1 = line.strip('\n').split('\t')
        data_list.append(tempo1)
    #temp2=shuffle(lines2)
    n_samples=len(data_list)
    data_array=np.array(data_list)
    
    with open(label_path, 'r') as f2:
        lines3=f2.readlines()
    label_list=[]
    for line in lines3:
        tempo2=line.strip('\n').split('\t')
        label_list.append(tempo2)
    
    label_array = np.array(label_list)
    label_array_no_id = label_array[:,1:2]
    combine=np.hstack((data_array,label_array_no_id))
    np.random.shuffle(combine)
    #print(combine2[0][0])
    
    n_training=int(0.8*n_samples)
    n_test = n_samples - n_training
    trainn=combine[0:n_training,:]
    testn=combine[n_training:,:]
    
    text_len_train=np.zeros((trainn.shape[0],1))
    text_len_test=np.zeros((testn.shape[0],1))
    for i in range(0,trainn.shape[0]):
        text_len_train[i]=trainn[i][1].count(' ')+1
    for i in range(0,testn.shape[0]):
        text_len_test[i]=testn[i][1].count(' ')+1
    #print(text_len_train[0][0])
    trainn=np.hstack((trainn,text_len_train))
    testn=np.hstack((testn,text_len_test))
    #print(trainn[0][6],trainn[1][6])
    trainn = trainn[trainn[:,6].argsort()]
    testn = testn[testn[:,6].argsort()]
    
    word_embedding_path = current_dir + 'resource/word_embeddings.json'
    word_dict = get_word2vec_dict(word_embedding_path)
    frequency_path = current_dir + 'resource/word_frequence'
    freqency_dict = get_freqency_dict(frequency_path)
    
    trainn = words2vecs(trainn, word_dict, freqency_dict)
    testn = words2vecs(testn, word_dict, freqency_dict)
    
    train_out_path = current_dir + 'processed_data/train_data.txt'
    trainfile = open(train_out_path, 'w', encoding='utf-8')
    json.dump(trainn, trainfile, indent = 4, ensure_ascii=False)
    test_out_path = current_dir + 'processed_data/test_data.txt'
    testfile = open(test_out_path, 'w', encoding='utf-8')
    json.dump(testn, testfile, indent = 4, ensure_ascii=False)
    
if __name__ == 'main':
    split_data()
    