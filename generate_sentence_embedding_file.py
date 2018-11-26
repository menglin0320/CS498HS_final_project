import json
import numpy as np
import pickle
from project_configuration import config
from nltk import tokenize, word_tokenize
from sklearn.decomposition import TruncatedSVD


# U, S, Vt = randomized_svd(X, n_components=1)
def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def PCA_trans(files):
    ind_list = []
    all_vecs = []
    for i, file in enumerate(files):
        for sentence_vec in file[0]:
            all_vecs.append(sentence_vec)
            ind_list.append(i)
    all_file_matrix = np.asarray(all_vecs).transpose()
    pc = compute_pc(all_file_matrix, npc=1)
    all_file_matrix = all_file_matrix - all_file_matrix.dot(pc.transpose()) * pc

    previous_ind = ind_list[0]
    cur_list = []

    for i, ind in enumerate(ind_list):
        if ind == previous_ind:
            cur_list.append(all_file_matrix[:, i])
        else:
            files[previous_ind][0] = cur_list
            cur_list = []
            cur_list.append(all_file_matrix[:, i])
            previous_ind = ind
    cur_list.append(all_file_matrix[:, i])
    files[previous_ind][0] = cur_list
    return files


def basic_sentence_split(words):
    sentences = []
    for i in range(0, len(words), 10):
        sentences.append(words[i:i + 10])
    return sentences


def sentence2vector(sentences, word_dict, freqency_dict):
    vectors = []
    total_words_appear = 1500000000
    a = 1e-4
    for s in sentences:
        s = s.replace('https:', '')
        s = s.replace('http:', '')
        words = word_tokenize(s)
        tmp = np.zeros(300)
        for word in words:
            try:
                # weight = a / (a + freqency_dict[word] / total_words_appear)
                # cur_embedding = np.asarray(word_dict[word])
                tmp += a / (a + freqency_dict[word] / total_words_appear) * np.asarray(word_dict[word])
            except:
                print('{} not on frequency_dict or word_dict, discarded'.format(word))
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
        sentences = tokenize.sent_tokenize(in_array[i][0])
        sentence_vectors = sentence2vector(sentences, word_dict, freqency_dict)
        output_list[i][0] = sentence_vectors
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

    weird_word = {}
    for word in word_dict:
        if word not in frequency_dict:
            # print(word)
            if word not in weird_word:
                weird_word[word] = 1
            else:
                weird_word[word] += 1
            frequency_dict[word] = 1
    print("weird word:{}".format(len(weird_word)))
    return frequency_dict


def split_data(config):
    data_path = config.raw_data_path
    with open(data_path, 'rb') as pkl_f:
        combine = pickle.load(pkl_f)

    combine = np.array(combine)
    np.random.shuffle(combine)
    n_samples = len(combine)

    n_training = int(0.8 * n_samples)
    trainn = combine[0:n_training, :]
    testn = combine[n_training:, :]
    print("Data splited")
    # print(trainn[0])

    # word_dict = get_word2vec_dict(word_embedding_path)
    word_dict_path = config.word_embedding_path
    with open(word_dict_path, 'rb') as pkl_f:
        word_dict = pickle.load(pkl_f)
    tmp_dict = {}
    for word, embedding in word_dict.items():
        embed_list = embedding.split(' ')
        embed_list = [float(elem) for elem in embed_list]
        tmp_dict[word] = embed_list
    word_dict = tmp_dict
    del tmp_dict

    frequency_path = config.word_frequence_json
    freqency_dict = get_freqency_dict(word_dict, frequency_path)

    print("freq, word_vec loaded")

    trainn = words2vecs(trainn, word_dict, freqency_dict)
    trainn = PCA_trans(trainn)
    testn = words2vecs(testn, word_dict, freqency_dict)
    testn = PCA_trans(testn)
    print("Word 2 Vec Done")
    text_len_train = np.zeros((len(trainn), 1))
    text_len_test = np.zeros((len(testn), 1))
    # need to verify
    for i in range(0, len(trainn)):
        text_len_train[i] = len(trainn[i][0]) + 1
    for i in range(0, len(testn)):
        text_len_test[i] = len(testn[i][0]) + 1
    # print(text_len_train[0][0])
    trainn = np.hstack((trainn, text_len_train))
    testn = np.hstack((testn, text_len_test))
    del text_len_train
    del text_len_test
    del combine
    print("Sort files down")

    # print(trainn[0][6],trainn[1][6])
    # print(trainn[:, 6].astype(np.float32))
    trainn = trainn[trainn[:, -1].astype(np.float32).argsort()]
    # for sample in trainn:
    #     print(sample[6])
    testn = testn[testn[:, -1].astype(np.float32).argsort()]
    train_out_path = config.train_data_path
    trainfile = open(train_out_path, 'wb')
    pickle.dump(trainn, trainfile)
    test_out_path = config.test_data_path
    trainfile.close()
    testfile = open(test_out_path, 'wb')
    pickle.dump(testn, testfile)
    testfile.close()

    # testfile = open(test_out_path, 'rb')
    # tmp = pickle.load(testfile)
    # print(tmp[0])


if __name__ == '__main__':
    split_data(config)