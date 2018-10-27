import tensorflow as tf
from src.util.DNN_util.zone_out_lstm import ZoneoutLSTMCell
import tensorflow.contrib.layers as layers

class zone_out_lstm_model():
    def __init__(self, word_embedding_size):
        self.dim_hidden = 300
        self.embedding_batch = tf.placeholder(tf.float32, [None, None, word_embedding_size])
        self.labels = tf.placeholder(tf.int32, [None, 1])
        self.mask = tf.placeholder(tf.float32, [None, None])
        self.is_train = tf.placeholder(tf.bool, [])
        self.seq_len = tf.shape(self.embedding_batch)[1]
        self.initial_lr = tf.constant(0.2, dtype=tf.float32)
        self.n_labels = 2
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(0.0)
        self.define_variables()
        self.build_model()

    def define_variables(self):
        self.ZLSTM = ZoneoutLSTMCell(self.dim_hidden, self.is_train)
        self.batch_size = tf.shape(self.embedding_batch)[0]
        self.w_2logit = tf.get_variable('w_2logit', shape=[self.dim_hidden, self.n_labels],
                                        initializer=self.weight_initializer)

        self.bias_2logit = tf.get_variable("bias_2logit", shape=[self.n_labels],
                                           initializer=self.bias_initializer)

        self.counter_dis = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    # def convert_label2one_hot(self):
    #     labels = self.labels[:,:]
    #     labels = tf.expand_dims(labels, 1)
    #     batch_range = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
    #     sparse = tf.concat([batch_range, labels], 1)
    #     onehot = tf.sparse_to_dense(sparse, tf.stack([self.batch_size, self.n_labels]), 1.0, 0.0)
    #     return onehot

    def one_iteration(self, state, i):
        out, state = self.ZLSTM(self.embedding_batch[:, i, :], state)
        logits = tf.matmul(out, self.w_2logit) + self.bias_2logit
        one_hot = tf.one_hot(self.labels[:, i], 2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = one_hot, logits = logits)
        loss = loss * self.mask[:, i]
        predict = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        # print(predict.get_shape())
        correct_preditions = tf.equal(predict, self.labels[:, i])
        correct_preditions = tf.cast(correct_preditions, tf.float32) * self.mask[:, i]
        return state, loss, correct_preditions

    def build_model(self):
        with tf.variable_scope('classifier'):
            zero_state = self.ZLSTM.zero_state(self.batch_size, dtype=tf.float32)
            state, loss, correct_preditions = self.one_iteration(zero_state, 0)
            total_loss = loss
            total_corrects = correct_preditions
            tf.get_variable_scope().reuse_variables()

            i = tf.constant(1)

            while_condition = lambda i, N1, N2, N3: tf.less(i, self.seq_len)

            def body(i, state, total_corrects, total_loss):
                state, loss, correct_preditions = self.one_iteration(state, i)
                total_corrects += correct_preditions
                total_loss += loss
                return [i + 1, state, total_corrects, total_loss]

            # do the loop
            [i, state, total_corrects, total_loss] = tf.while_loop(while_condition, body, [i, state, total_corrects, loss])

        self.total_corrects = total_corrects
        self.loss = total_loss
        self.loss = tf.reduce_mean(self.loss)
        self.accuracy = tf.reduce_mean(self.total_corrects)
        self.lr = tf.train.exponential_decay(self.initial_lr, self.counter_dis, 30000, 0.96, staircase=True)
        self.opt = layers.optimize_loss(loss=self.loss, learning_rate=self.lr,
                                        optimizer=tf.train.AdadeltaOptimizer,
                                        clip_gradients=100., global_step=self.counter_dis)
