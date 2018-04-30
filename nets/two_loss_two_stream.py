import tensorflow as tf
import numpy as np
import constants as const
import configuration as file_const
from tensorflow.python import pywrap_tensorflow
from utils.logger import root_logger as logger

class TwoLossTwoStreamNet:
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


        if assign_weights:
            assign_operations = []
            assign_operations.append(conv1W.assign(net_data["conv1"][0]))
            assign_operations.append(conv1b.assign(net_data["conv1"][1]))
            assign_operations.append(conv2W.assign(net_data["conv2"][0]))
            assign_operations.append(conv2b.assign(net_data["conv2"][1]))
            assign_operations.append(conv3W.assign(net_data["conv3"][0]))
            assign_operations.append(conv3b.assign(net_data["conv3"][1]))
            assign_operations.append(conv4W.assign(net_data["conv4"][0]))
            assign_operations.append(conv4b.assign(net_data["conv4"][1]))
            assign_operations.append(conv5W.assign(net_data["conv5"][0]))
            assign_operations.append(conv5b.assign(net_data["conv5"][1]))

            self.assign_operations = assign_operations;


        return [conv1,conv2,conv3,conv4,conv5,fc6]

    def rgb_2_bgr(self,rgb):
        #rgb_scaled = rgb * 255.0
        rgb_scaled = rgb # I already fit it [0-255]

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
        bgr = tf.concat(axis=3, values=[
            blue - IMAGENET_MEAN_BGR[0],
            green - IMAGENET_MEAN_BGR[1],
            red - IMAGENET_MEAN_BGR[2],
        ])

        #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        return bgr;

    def print_tensors_in_checkpoint_file(self,file_name, tensor_name, all_tensors,
                                         all_tensor_names):
        """Prints tensors in a checkpoint file.
        If no `tensor_name` is provided, prints the tensor names and shapes
        in the checkpoint file.
        If `tensor_name` is provided, prints the content of the tensor.
        Args:
          file_name: Name of the checkpoint file.
          tensor_name: Name of the tensor in the checkpoint file to print.
          all_tensors: Boolean indicating whether to print all tensors.
          all_tensor_names: Boolean indicating whether to print all tensor names.
        """
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            if all_tensors or all_tensor_names:
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in sorted(var_to_shape_map):
                    print("tensor_name: ", key, np.mean(reader.get_tensor(key)))
                    if all_tensors:
                        print(reader.get_tensor(key))
            elif not tensor_name:
                print(reader.debug_string().decode("utf-8"))
            else:
                print("tensor_name: ", tensor_name)
                print(reader.get_tensor(tensor_name))
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")
            if ("Data loss" in str(e) and
                    (any([e in file_name for e in [".index", ".meta", ".data"]]))):
                proposed_file = ".".join(file_name.split(".")[0:-1])
                v2_file_error_template = """
    It's likely that this is a V2 checkpoint and you need to provide the filename
    *prefix*.  Try removing the '.' and extension.  Try:
    inspect checkpoint --file_name = {}"""

            print(v2_file_error_template.format(proposed_file))

    def load_pretrained(self,sess, ckpt_file):

        ## When I swicth from unsupervised learning to supervised learning, no supervised layer weights in the model.
        ## So load the trained layer weights are stored while supervised layer weights are ignored
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
        ckpt_vars = reader.get_variable_to_shape_map()
        ckpt_variables = ckpt_vars.keys()
        common_variables = [];

        ignore_list = ['supervised_fc/fc7/kernel','supervised_fc/fc7/bias',
                       'supervised_fc/fc8/kernel','supervised_fc/fc8/bias','supervised_fc/logits/kernel',
                       'supervised_fc/logits/bias',

                       'unsupervised_fc/fc7/kernel', 'unsupervised_fc/fc7/bias',
                       'unsupervised_fc/fc8/kernel', 'unsupervised_fc/fc8/bias', 'unsupervised_fc/logits/kernel',
                       'unsupervised_fc/logits/bias',

                       'siamese/word_fc6/fc6b','siamese/word_fc6/fc6W','siamese/word_fc6/fc6b/Adam_1',
                       'siamese/word_fc6/fc6b/Adam','siamese/word_fc6/fc6W/Adam_1','siamese/word_fc6/fc6W/Adam',

                       'siamese/cntxt_fc6/fc6W','siamese/cntxt_fc6/fc6b','siamese/cntxt_fc6/fc6b/Adam_1',
                       'siamese/cntxt_fc6/fc6b/Adam','siamese/cntxt_fc6/fc6W/Adam_1','siamese/cntxt_fc6/fc6W/Adam',
                       ]
        for v_old in ckpt_variables:
            for v in all_variables:
                if(v.name[:-2] == v_old):
                    #if(v_old == 'supervised_fc/supervised_fc_256/bias' or v_old == 'supervised_fc/supervised_fc_256/kernel' or
                    #           v_old == 'supervised_fc/supervised_fc/kernel' or v_old == 'supervised_fc/supervised_fc/bias'):
                    found = [1 for tmp in ignore_list if v_old == tmp  or v_old.startswith(tmp)]
                    if(len(found) > 0):
                        break;
                    else:
                        common_variables.append(v);
                        break;

        print('Loaded variables ',common_variables)
        temp_saver = tf.train.Saver(var_list=common_variables)
        temp_saver.restore(sess, ckpt_file)



    def print_means(self,sess):
        vars = [v for v in tf.trainable_variables()]
        for var in vars:
            var_value =sess.run(var)
            print(var_value.shape,np.mean(var_value))
        print('===================================================')


    def inspect_output(self,sess,feeddict):
        word_dense, context_dense = sess.run([self.fcf_word_dense,self.fcf_context_dense], feeddict)
        print(np.mean(word_dense),np.mean(context_dense ))


    def __init__(self,mode,train_spatial_tower=False,train_motion_tower=True,load_alex_weights=False):
        net_data = np.load(open(file_const.model_weights_filepath, "rb"), encoding="latin1").item()

        batch_size = None
        self.input_words = tf.placeholder(tf.float32,
                                          shape=(
                                          batch_size, const.frame_height, const.frame_width, const.frame_channels),
                                          name='words_input')

        self.input_context = tf.placeholder(tf.float32,
                                            shape=(
                                            batch_size, const.frame_height, const.frame_width, const.context_channels),
                                            name='context_input')
        self.labels_balance = tf.placeholder(tf.float32, shape=(batch_size),
                                             name='labels_balance')  # balanced = 1 for supervised , balanced = 0 for unsupervised
        num_classes = file_const.num_classes
        unsupervised_num_classes = file_const.unsupervised_num_classes
        self.supervised_labels = tf.placeholder(tf.int32, shape=(batch_size, num_classes), name='class_lbls')
        self.unsupervised_labels = tf.placeholder(tf.int32, shape=(batch_size, unsupervised_num_classes ), name='gt_lbls')

        words = self.rgb_2_bgr(self.input_words)
        context = self.input_context

        fc6_num_units = 4096
        with tf.variable_scope("siamese",reuse=tf.AUTO_REUSE) as scope:

            self.layers1 = self.build_net(words, net_data,train_params=train_spatial_tower,prefix='word_',assign_weights=load_alex_weights,fc6_num_units=fc6_num_units )
            self.fcf_word_dense= self.layers1[-1]
            #fc8 = self.layers1[-1]


            self.layers2 =  self.build_net(context, net_data,train_params=train_motion_tower,prefix='cntxt_',fc6_num_units=fc6_num_units )
            self.fcf_context_dense = self.layers2[-1];

        ## Change fusion layer to be Addition instead of subtraction
        fusion_layer = self.fcf_word_dense + self.fcf_context_dense



        ## The weights of the following FC layers are still <<<<<< OPEN QUESTION >>>>>>
        supervised = True
        num_units = 4096
        drop_out_rate = file_const.dropout_rate;
        logger.info('drop_out_rate '+str(drop_out_rate ))
        ## *********************** Supervised **********************##
        with tf.variable_scope("supervised_fc") as scope:
            supervised_fc6_dropout = tf.layers.dropout(fusion_layer, rate=drop_out_rate ,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)
            supervised_fc7 = tf.layers.dense(inputs=supervised_fc6_dropout , units=num_units , name='fc7', activation=tf.nn.relu,trainable=supervised);
            supervised_fc7_dropout = tf.layers.dropout(inputs=supervised_fc7 , rate=drop_out_rate ,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)
            supervised_fc8 = tf.layers.dense(inputs=supervised_fc7_dropout, units=num_units, name='fc8',
                                             activation=tf.nn.relu, trainable=supervised);
            supervised_logits = tf.layers.dense(inputs=supervised_fc8 , units=num_classes, name='logits',trainable=supervised)

        with tf.variable_scope("supervised_loss") as scope:
            supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.supervised_labels, logits=supervised_logits, name='xentropy')
            self.supervised_loss = tf.reduce_mean(self.labels_balance * supervised_cross_entropy, name='xentropy_mean')
            self.supervised_logits = tf.nn.softmax(supervised_logits)


        with tf.name_scope('supervised_accuracy'):
            self.supervised_class_prediction = tf.argmax(supervised_logits, 1)
            supervised_correct_prediction = tf.equal(tf.argmax(self.supervised_labels, 1), self.supervised_class_prediction )
            self.supervised_correct_prediction = tf.cast(supervised_correct_prediction, tf.float32)
            self.supervised_accuracy = tf.reduce_mean(self.supervised_correct_prediction)

        num_units = 128
        supervised = False
        ## *********************** Unsupervised **********************##
        with tf.variable_scope("unsupervised_fc") as scope:
            unsupervised_fc7 = tf.layers.dense(inputs=fusion_layer, units=num_units, name='fc7', activation=tf.nn.relu,
                                               trainable=not supervised);
            unsupervised_fc8 = tf.layers.dense(inputs=unsupervised_fc7, units=num_units, name='fc8',
                                               activation=tf.nn.relu, trainable=not supervised);
            unsupervised_logits = tf.layers.dense(inputs=unsupervised_fc8, units=unsupervised_num_classes,
                                                  name='logits', trainable=not supervised)


        with tf.variable_scope("unsupervised_loss") as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.unsupervised_labels, logits=unsupervised_logits , name='xentropy')
            self.unsupervised_loss = file_const.unsup_weight * tf.reduce_mean((1 - self.labels_balance) * cross_entropy, name='xentropy_mean')
            self.unsupervised_logits = tf.nn.softmax(unsupervised_logits )


        with tf.name_scope('unsupervised_accuracy'):
            unsupervised_correct_prediction = tf.equal(tf.argmax(self.unsupervised_labels, 1), tf.argmax(unsupervised_logits , 1))
            self.unsupervised_correct_prediction = tf.cast(unsupervised_correct_prediction , tf.float32)
            self.unsupervised_accuracy = tf.reduce_mean(self.unsupervised_correct_prediction)


        self.loss =  self.supervised_loss + self.unsupervised_loss;
        self.accuracy = self.labels_balance * self.unsupervised_accuracy + (1-self.labels_balance) * self.supervised_accuracy
