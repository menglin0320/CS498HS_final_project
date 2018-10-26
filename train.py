from src.util.file_util import *
from project_configuration import *
from src.get_batches import get_batches
from model.zone_out_lstm_model import *
import os
import tensorflow as tf
import numpy as np


def initialize_model(checkpoint_dir):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    model = zone_out_lstm_model()
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


if __name__ == '__main__':
    data_path = config.data_path
    batch_size = config.batch_size
    checkpoint_dir = config.checkpoint_dir

    n_epoch = 100000
    # TODO write the valid read data function
    data_dict = load_data(data_path)
    data_train_batches = get_batches(data_dict['train'], batch_size)
    data_test_batches = get_batches(data_dict['test'], batch_size)

    model, saver, sess, start_step = initialize_model(checkpoint_dir)
    n_batches = data_train_batches[0]
    for i in range(start_step // n_batches, n_epoch):
        rand_permute = np.arange(batch_size)
        np.random.shuffle(rand_permute)
        saver.save(sess, checkpoint_dir, global_step=i * rand_permute.shape[0])
        train_avg_loss = 0
        train_avg_accy = 0
        test_avg_loss = 0
        test_avg_accy = 0
        count = 0
        for j in range(0, rand_permute.shape[0]):
            cur_batch = data_train_batches[rand_permute[j]]
            _, train_loss, train_accy = sess.run([model.opt, model.loss, model.accuracy],
                                                 feed_dict={model.embedding_batch: cur_batch['data'],
                                                            model.labels: cur_batch['label'],
                                                            model.mask: cur_batch['mask'],
                                                            model.is_train: True})
            train_avg_loss += train_loss
            train_avg_accy += train_accy

        for j in range(0, data_test_batches.shape[0]):
            cur_batch = data_test_batches[j]
            test_loss, test_accy = sess.run([model.loss, model.accuracy],
                                                 feed_dict={model.embedding_batch: cur_batch['data'],
                                                            model.labels: cur_batch['label'],
                                                            model.mask: cur_batch['mask'],
                                                            model.is_train: False})
            test_avg_loss += test_loss
            test_avg_accy += test_accy

        print('for batch {}: on training_sample avg loss is {}, avg_accy is {}'.format(i, train_avg_loss, train_avg_accy))
        print('for batch {}: on test_sample avg loss is {}, avg_accy is {}'.format(i, test_avg_loss, test_avg_accy))