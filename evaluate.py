from src.util.file_util import *
from project_configuration import *
from src.get_batches import get_batches
from model.zone_out_lstm_model import *
import os
import tensorflow as tf
import numpy as np
import pickle
import sys

def bool2int(b):
    if b == 'True':
        return 1
    return 0

def initialize_model(checkpoint_dir):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    model = zone_out_lstm_model(300)
    saver = tf.train.Saver(max_to_keep=10)
    saved_path = tf.train.latest_checkpoint(checkpoint_dir)
    start_step = 0
    if (saved_path):
        # tf.reset_default_graph()
        saver.restore(sess, saved_path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        start_step = int(step)
        print('model restored', start_step)
    else:
        sess.run(tf.global_variables_initializer())
    return model, saver, sess, start_step


def load_data(train_path, test_path):
    with open(train_path, 'rb') as f:
        train_data_raw = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data_raw = pickle.load(f)
    full_data = {}
    full_data['train'] = []
    full_data['test'] = []
    for i, cell in enumerate(train_data_raw):
        full_data['train'].append({})
        full_data['train'][i]['attributes'] = cell[0]
        # print('real_length{} and leangth used for sorting {}'.format(len(cell[1]),cell[6]))
        full_data['train'][i]['label'] = bool2int(cell[1])
        full_data['train'][i]['timestamp'] = float(cell[2])

    for i, cell in enumerate(test_data_raw):
        full_data['test'].append({})
        full_data['test'][i]['attributes'] = cell[0]
        full_data['test'][i]['label'] = bool2int(cell[1])
        full_data['test'][i]['timestamp'] = float(cell[2])

    return full_data



def evaluate(data_path):
    batch_size = config.batch_size
    checkpoint_dir = config.checkpoint_dir
    train_path = config.train_data_path
    test_path = data_path
    # TODO write the valid read data function
    data_dict = load_data(train_path, test_path)
    data_train_batches = get_batches(data_dict['train'], batch_size)
    data_test_batches = get_batches(data_dict['test'], batch_size)

    model, saver, sess, start_step = initialize_model(checkpoint_dir)
    n_batches = len(data_train_batches)
    predict_label_pairs = np.zeros((len(data_dict['test']), 2))
    for j in range(0, len(data_test_batches)):
        cur_batch = data_test_batches[j]
        predicts = sess.run([model.predicts],
                                        feed_dict={model.embedding_batch: cur_batch['data'],
                                                   model.labels: cur_batch['label'],
                                                   model.mask: cur_batch['mask'],
                                                   model.time_stamp: cur_batch['timestamp'],
                                                   model.is_train: False})

        for i, predict in enumerate(predicts[0]):
            predict_label_pairs[j * batch_size + i][0] = predict
            predict_label_pairs[j * batch_size + i][1] = cur_batch['label'][i]
    label_ratio = np.sum(predict_label_pairs[:,1])/predict_label_pairs.shape[0]
    print('label_ratio is {}'.format(label_ratio))
    accuracy = np.mean(np.equal(predict_label_pairs[:,0], predict_label_pairs[:,1]))
    print ('accuracy is {}'.format(accuracy))

    correct_predict_inds = predict_label_pairs[:,0] == 1
    correct_predict_inds = np.tile(np.expand_dims(correct_predict_inds, 1), (1, 2))
    correct_predict_samples = np.reshape(predict_label_pairs[correct_predict_inds],(-1, 2))
    precision = np.mean(np.equal(correct_predict_samples[:,0], correct_predict_samples[:,1]))
    print('precision is : {}'.format(precision))

    correct_label_inds = predict_label_pairs[:,1] == 1
    correct_label_inds = np.tile(np.expand_dims(correct_label_inds, 1), (1, 2))
    correct_label_samples = np.reshape(predict_label_pairs[correct_label_inds],(-1,2))
    recall = np.mean(np.equal(correct_label_samples[:,0], correct_label_samples[:,1]))
    print('recall is : {}'.format(recall))

    with open('output.txt', 'w') as f:
        for pair in predict_label_pairs:
            f.write(pair[0] + '\n')

    sess.close()
    return accuracy, precision, recall



if __name__ == '__main__':
    if len(sys.argv) != 2:
        # print(sys.argv)
        raise ValueError('please give 1 arg to specify k')
    pickle_file_path = sys.argv[1]
    evaluate(pickle_file_path)