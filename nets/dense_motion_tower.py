import os
import sys
import tensorflow as tf
import numpy as np
import constants as const
import configuration as file_const
from tensorflow.python import pywrap_tensorflow

class MotionTower:
    def conv(self,input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group == 1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
    def build_net(self,input,net_data,train_params=True,prefix='',assign_weights=False,fc6_num_units = 128):

        if (prefix == 'cntxt_'):
            c_i = const.context_channels
        else:
            c_i = const.frame_channels

        trainable_params = train_params
        with tf.variable_scope(prefix+"conv1"):

            kernel_size = 11;
            no_filters = 96;
            stride_size = 4;
            conv1W = tf.get_variable(name='conv1W',shape=(kernel_size,kernel_size,c_i,no_filters),trainable=trainable_params)
            conv1b = tf.get_variable(name='conv1b',shape=net_data["conv1"][1].shape,trainable=trainable_params)

            conv1_in = self.conv(input, conv1W, conv1b, kernel_size, kernel_size, no_filters, stride_size, stride_size , padding="SAME", group=1)
            conv1 = tf.nn.relu(conv1_in)
            # lrn1
            # lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            kernel_size = 3;
            k_w = 3;
            stride_size = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, kernel_size, k_w, 1], strides=[1, stride_size, s_w, 1], padding=padding)

        with tf.variable_scope(prefix+"conv2"):
            # conv2
            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            kernel_size = 5;
            no_filters = 256;
            stride_size = 1;
            group = 2
            #conv2W = tf.Variable(net_data["conv2"][0])
            conv2W = tf.get_variable(name='conv2W', shape=net_data["conv2"][0].shape, trainable=trainable_params)

            #conv2b = tf.get_variable(net_data["conv2"][1])
            conv2b = tf.get_variable(name='conv2b', shape=net_data["conv2"][1].shape, trainable=trainable_params)

            conv2_in = self.conv(maxpool1, conv2W, conv2b, kernel_size, kernel_size, no_filters, stride_size, stride_size, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in)

            # lrn2
            # lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            # maxpool2
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            kernel_size = 3;
            stride_size = 2;
            padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, kernel_size, kernel_size , 1], strides=[1, stride_size, stride_size, 1], padding=padding)

        with tf.variable_scope(prefix+"conv3"):
            # conv3
            # conv(3, 3, 384, 1, 1, name='conv3')
            kernel_size = 3;
            no_filters = 384;
            stride_size = 1;
            group = 1
            # conv3W = tf.Variable(net_data["conv3"][0])
            conv3W = tf.get_variable(name='conv3W',shape=net_data["conv3"][0].shape,trainable=trainable_params)

            # conv3b = tf.Variable(net_data["conv3"][1])
            conv3b = tf.get_variable(name='conv3b',shape=net_data["conv3"][1].shape,trainable=trainable_params)

            conv3_in = self.conv(maxpool2, conv3W, conv3b, kernel_size, kernel_size, no_filters, stride_size, stride_size, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

        with tf.variable_scope(prefix+"conv4"):
            # conv4
            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            kernel_size = 3;
            no_filters = 384;
            stride_size = 1;
            group = 2
            # conv4W = tf.Variable(net_data["conv4"][0])
            conv4W = tf.get_variable(name='conv4W',shape=net_data["conv4"][0].shape,trainable=trainable_params);
            # conv4b = tf.Variable(net_data["conv4"][1])
            conv4b = tf.get_variable(name='conv4b',shape=net_data["conv4"][1].shape,trainable=trainable_params)

            conv4_in = self.conv(conv3, conv4W, conv4b, kernel_size, kernel_size, no_filters, stride_size, stride_size, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)
        with tf.variable_scope(prefix+"conv5"):
            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            kernel_size = 3;
            no_filters = 256;
            stride_size = 1;
            group = 2
            # conv5W = tf.Variable(net_data["conv5"][0])
            # conv5b = tf.Variable(net_data["conv5"][1])
            conv5W = tf.get_variable(name='conv5W',shape=net_data["conv5"][0].shape,trainable=trainable_params);
            conv5b = tf.get_variable(name='conv5b',shape=net_data["conv5"][1].shape,trainable=trainable_params)

            conv5_in = self.conv(conv4, conv5W, conv5b, kernel_size, kernel_size, no_filters, stride_size, stride_size, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            kernel_size = 3;
            stride_size = 2;
            padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)

        with tf.variable_scope(prefix+"fc6"):
            # fc6
            # fc6W = tf.Variable(net_data["fc6"][0])
            # fc6b = tf.Variable(net_data["fc6"][1])
            fc6W = tf.get_variable(name='fc6W',shape=(9216, fc6_num_units),trainable=True) ## Settings trainable to false is nonsense
            fc6b = tf.get_variable(name='fc6b', shape=(fc6_num_units,), trainable=True) ## Settings trainable to false is nonsense


            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        return [conv1, conv2, conv3, conv4, conv5,fc6]

    def load_pretrained(self,sess, ckpt_file):
        ## Load pre-trained conv layer variables
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        common_variables = [];
        for v in all_variables:
            if v.name.startswith('siamese/cntxt_') and 'fc6' not in v.name:
                common_variables.append(v)

        print('Loaded variables ', common_variables)
        temp_saver = tf.train.Saver(var_list=common_variables)
        temp_saver.restore(sess, ckpt_file)

    def load_weights(self,sess,saver,ckpt_file,save_model_dir,load_alex_weights):
        print('Model Path ', ckpt_file)
        if (os.path.exists(save_model_dir) and len(os.listdir(save_model_dir)) > 1):
            try:
                # Try to restore everything if possible
                saver.restore(sess, ckpt_file)
                print('Model Loaded Normally');
            except:
                ## If not, load as much as possible
                self.load_pretrained(sess, ckpt_file);
                print('Pretrained Weights loaded, while some layers are randomized')

        elif load_alex_weights:
            print('Loading img2vec_model.assign_operations:', len(self.assign_operations));
            sess.run(self.assign_operations);


    def __init__(self, mode = tf.estimator.ModeKeys.TRAIN,train_motion_tower=True):
        net_data = np.load(open(file_const.model_weights_filepath, "rb"), encoding="latin1").item()
        batch_size = None
        self.input_context = tf.placeholder(tf.float32,
                                            shape=(
                                                batch_size, const.frame_height, const.frame_width,
                                                const.context_channels),
                                            name='context_input')

        num_classes = file_const.num_classes
        self.supervised_labels = tf.placeholder(tf.int32, shape=(batch_size, num_classes), name='class_lbls')

        context = self.input_context
        fcn_num_units = 4096

        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            self.layers2 = self.build_net(context, net_data, train_params=train_motion_tower, prefix='cntxt_',
                                          fc6_num_units=fcn_num_units)
            self.fc6_context_dense = self.layers2[-1];


        ## *********************** Supervised **********************##
        with tf.variable_scope("supervised_fc") as scope:
            supervised_fc7 = tf.layers.dense(inputs=self.fc6_context_dense , units=fcn_num_units, name='fc7', activation=tf.nn.relu,trainable=True);
            supervised_fc8 = tf.layers.dense(inputs=supervised_fc7 , units=fcn_num_units, name='fc8',
                                             activation=tf.nn.relu, trainable=True);

            supervised_logits = tf.layers.dense(inputs=supervised_fc8 , units=num_classes, name='logits',trainable=True)

        with tf.variable_scope("supervised_loss") as scope:
            supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.supervised_labels,
                                                                               logits=supervised_logits,
                                                                               name='xentropy')
            self.supervised_logits = tf.nn.softmax(supervised_logits)
            self.supervised_loss = tf.reduce_mean(supervised_cross_entropy, name='xentropy_mean')

        with tf.name_scope('supervised_accuracy'):
            self.class_prediction = tf.argmax(supervised_logits, 1)
            supervised_correct_prediction = tf.equal(tf.argmax(self.supervised_labels, 1), self.class_prediction)
            self.supervised_correct_prediction = tf.cast(supervised_correct_prediction, tf.float32)
            self.supervised_accuracy = tf.reduce_mean(self.supervised_correct_prediction)





