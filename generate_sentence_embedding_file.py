import json
import numpy as np
import os
import pickle
from project_configuration import config

def basic_sentence_split(words):
    sentences = []
    for i in range(0, len(words), 10):
        sentences.append(words[i:i + 10])
    return sentences


def sentence2vector(sentences, word_dict, freqency_dict):
    vectors = []
    total_words_appear = 1500000000
    a = 10e-3
    for s in sentences:
        tmp = np.zeros(300)
        for word in s:
            tmp += a / (a + freqency_dict[word]/total_words_appear) * np.asarray(word_dict[word])
        tmp /= len(s)
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
        word_dict[w] = [float(num_str) for num_str in word_dict[w].split(' ')]
    return word_dict


def get_freqency_dict(word_dict, frequency_path=""):
    if frequency_path == "":
        return get_dummy_dict(word_dict)
    frequency_dict = {}
    with open(frequency_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(' ')
        if (line[0] in word_dict):
            frequency_dict[line[0]] = float(line[1])

    for word in word_dict:
        if word not in frequency_dict:
            print(word)
            frequency_dict[word]=1
    return frequency_dict


def split_data(config):
    print("In split_data()")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current Dir:" + current_dir)
    sample_path = config.processed_data_path
    label_path = config.label_path
    print(sample_path)
    print(label_path)

    with open(sample_path, 'r') as f1:
        lines = f1.readlines()
    data_list = []
    for line in lines:
        tempo1 = line.strip('\n').split('\t')
        data_list.append(tempo1)
    # temp2=shuffle(lines2)
    n_samples = len(data_list)
    data_array = np.array(data_list)

    print("Attribute Read Done")

    with open(label_path, 'r') as f2:
        lines3 = f2.readlines()
    label_list = []
    for line in lines3:
        tempo2 = line.strip('\n').split('\t')
        label_list.append(tempo2)

    label_array = np.array(label_list)
    label_array_no_id = label_array[:, 1:2]
    combine = np.hstack((data_array, label_array_no_id))
    np.random.shuffle(combine)
    print("Label Added")
    del label_array
    del label_array_no_id
    del data_array
    # print(combine2[0][0])

    n_training = int(0.8 * n_samples)
    n_test = n_samples - n_training
    trainn = combine[0:n_training, :]
    testn = combine[n_training:, :]

    text_len_train = np.zeros((trainn.shape[0], 1))
    text_len_test = np.zeros((testn.shape[0], 1))
    for i in range(0, trainn.shape[0]):
        text_len_train[i] = trainn[i][1].count(' ') + 1
    for i in range(0, testn.shape[0]):
        text_len_test[i] = testn[i][1].count(' ') + 1
    # print(text_len_train[0][0])
    trainn = np.hstack((trainn, text_len_train))
    testn = np.hstack((testn, text_len_test))
    del text_len_train
    del text_len_test
    del combine
    # print(trainn[0][6],trainn[1][6])
    trainn = trainn[trainn[:, 6].argsort()]
    testn = testn[testn[:, 6].argsort()]

    print("Data split Done")
    print(trainn[0])
    word_embedding_path = current_dir + '/resource/word_embeddings.json'
    word_dict = get_word2vec_dict(word_embedding_path)
    frequency_path = current_dir + '/resource/relevant_corpus2.txt'
    freqency_dict = get_freqency_dict(word_dict, frequency_path)

    trainn = words2vecs(trainn, word_dict, freqency_dict)
    testn = words2vecs(testn, word_dict, freqency_dict)

    print("Word 2 Vec Done")

    train_out_path = config.train_data_path
    trainfile = open(train_out_path, 'wb')
    pickle.dump(trainn, trainfile)
    test_out_path = config.test_data_path
    trainfile.close()
    testfile = open(test_out_path, 'wb')
    pickle.dump(testn, testfile)
    testfile.close()
    
    #testfile = open(test_out_path, 'rb')
    #tmp = pickle.load(testfile)
    #print(tmp[0])

if __name__ == '__main__':
    # print("Hello!")
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # trainn = np.array([['17995', 'thought', '1952', 'A', '0.24', '0', '1.0']])
    # word_embedding_path = current_dir + '/resource/word_embeddings.json'
    # word_dict = get_word2vec_dict(word_embedding_path)
    # frequency_path = current_dir + '/resource/relevant_corpus2.txt'
    # freqency_dict = get_freqency_dict(word_dict, frequency_path)
    #
    # trainn = words2vecs(trainn, word_dict, freqency_dict)
    split_data(config)
