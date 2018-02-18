import tensorflow as tf
import numpy as np
import constants as const
import file_constants as file_const
from tensorflow.python import pywrap_tensorflow
from datetime import datetime


class OddOneOutNet:
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


    def build_net(self,input,net_data,train_params=True,prefix='',assign_weights=False):

        if (prefix == 'cntxt_'):
            c_i = const.context_channels
        else:
            c_i = const.frame_channels

        trainable_params = train_params
        with tf.variable_scope(prefix+"conv1"):
            k_h = 11;
            k_w = 11;


            c_o = 96;
            s_h = 4;
            s_w = 4

            conv1W = tf.get_variable(name='conv1W',shape=(k_h,k_h,c_i,c_o),trainable=trainable_params)

            conv1b = tf.get_variable(name='conv1b',shape=net_data["conv1"][1].shape,trainable=trainable_params)

            conv1_in = self.conv(input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
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

            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(prefix+"conv2"):
            # conv2
            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5;
            k_w = 5;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            #conv2W = tf.Variable(net_data["conv2"][0])
            conv2W = tf.get_variable(name='conv2W', shape=net_data["conv2"][0].shape, trainable=trainable_params)

            #conv2b = tf.get_variable(net_data["conv2"][1])
            conv2b = tf.get_variable(name='conv2b', shape=net_data["conv2"][1].shape, trainable=trainable_params)

            conv2_in = self.conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
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
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(prefix+"conv3"):
            # conv3
            # conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 1
            # conv3W = tf.Variable(net_data["conv3"][0])
            conv3W = tf.get_variable(name='conv3W',shape=net_data["conv3"][0].shape,trainable=trainable_params)

            # conv3b = tf.Variable(net_data["conv3"][1])
            conv3b = tf.get_variable(name='conv3b',shape=net_data["conv3"][1].shape,trainable=trainable_params)

            conv3_in = self.conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

        with tf.variable_scope(prefix+"conv4"):
            # conv4
            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 2
            # conv4W = tf.Variable(net_data["conv4"][0])
            conv4W = tf.get_variable(name='conv4W',shape=net_data["conv4"][0].shape,trainable=trainable_params);
            # conv4b = tf.Variable(net_data["conv4"][1])
            conv4b = tf.get_variable(name='conv4b',shape=net_data["conv4"][1].shape,trainable=trainable_params)

            conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)
        with tf.variable_scope(prefix+"conv5"):
            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3;
            k_w = 3;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            # conv5W = tf.Variable(net_data["conv5"][0])
            # conv5b = tf.Variable(net_data["conv5"][1])
            conv5W = tf.get_variable(name='conv5W',shape=net_data["conv5"][0].shape,trainable=trainable_params);
            conv5b = tf.get_variable(name='conv5b',shape=net_data["conv5"][1].shape,trainable=trainable_params)

            conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(prefix+"fc6"):
            fc6_num_units = 128
            # fc6
            # fc6W = tf.Variable(net_data["fc6"][0])
            # fc6b = tf.Variable(net_data["fc6"][1])
            fc6W = tf.get_variable(name='fc6W',shape=(9216, fc6_num_units),trainable=trainable_params)
            fc6b = tf.get_variable(name='fc6b', shape=(fc6_num_units,), trainable=trainable_params)


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

        for v_old in ckpt_variables:
            for v in all_variables:
                if(v.name[:-2] == v_old):
                    if(v_old == 'supervised_fc/supervised_fc_256/bias' or v_old == 'supervised_fc/supervised_fc_256/kernel' or
                               v_old == 'supervised_fc/supervised_fc/kernel' or v_old == 'supervised_fc/supervised_fc/bias'):
                        break;
                    else:
                        common_variables.append(v);
                        break;


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


    def __init__(self,load_alex_weights=False,num_clips=2):
        net_data = np.load(open(file_const.model_weights_filepath, "rb"), encoding="latin1").item()

        batch_size = None

        self.input_context = tf.placeholder(tf.float32,
                                            shape=(
                                            batch_size, const.frame_height, const.frame_width, const.context_channels * num_clips * 2),
                                            name='context_input')


        self.unsupervised_labels = tf.placeholder(tf.int32, shape=(batch_size, num_clips), name='gt_lbls')

        context = self.input_context

        clips_embedding = []
        smooth_clips_embedding = []
        shift = num_clips * const.context_channels
        with tf.variable_scope("Smooth_O3N",reuse=tf.AUTO_REUSE) as scope:
            for i in range(num_clips):
                if(i == 0):
                    layers = self.build_net(context[:,:,:,i*const.context_channels:(i+1)*(const.context_channels)], net_data,train_params=True,prefix='cntxt_')
                    clip_embedding =layers[-1];
                else:

                    layers =  self.build_net(context[:,:,:,i*const.context_channels:(i+1)*(const.context_channels)], net_data,train_params=True,prefix='cntxt_')
                    clip_embedding = layers[-1];

                clips_embedding.append(clip_embedding);

            for i in range(num_clips):
                if (i == 0):
                    layers = self.build_net(context[:,:,:,shift+i*const.context_channels:shift +(i+1)*(const.context_channels)], net_data, train_params=True, prefix='cntxt_')
                    smooth_clip_embedding = layers[-1];
                else:
                    layers = self.build_net(context[:,:,:,shift +i*const.context_channels:shift +(i+1)*(const.context_channels)], net_data, train_params=True, prefix='cntxt_')
                    smooth_clip_embedding  = layers[-1];

                smooth_clips_embedding.append(smooth_clip_embedding);

        ## Sum of diff

        for i in range(num_clips):
            for j in range(i+1,num_clips):
                diff = clips_embedding[i] - clips_embedding[j]
                if(i ==0 and j==1): # first sub
                    self.sum_of_diff = diff ;
                else:
                    self.sum_of_diff += diff;

        ## *********************** O3N FCN  **********************##
        with tf.variable_scope("O3N_FCN") as scope:
            unsupervised_fc7 = tf.layers.dense(inputs=self.sum_of_diff, units=128, name='F7_128',activation=tf.nn.relu);
            unsupervised_fc8 = tf.layers.dense(inputs=unsupervised_fc7, units=128, name='F8_128',activation=tf.nn.relu);
            unsupervised_logits = tf.layers.dense(inputs=unsupervised_fc8, units=num_clips, name='fc_prediction')

        with tf.variable_scope("O3N_loss") as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.unsupervised_labels,
                                                                    logits=unsupervised_logits, name='xentropy')
            self.logits = tf.nn.softmax(unsupervised_logits)
            loss1 = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        with tf.name_scope('O3N_accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.unsupervised_labels, 1),
                                          tf.argmax(unsupervised_logits, 1))
            self.correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        with tf.variable_scope("smooth_loss") as scope:
            for i in range(num_clips):
                if(i ==0):
                    loss2 = tf.reduce_sum(tf.norm(clips_embedding[i],axis=1) - tf.norm(smooth_clips_embedding[i],axis=1)) ## May be reduce mean better?
                else:
                    loss2 += tf.reduce_sum(tf.norm(clips_embedding[i], axis=1) - tf.norm(smooth_clips_embedding[i],axis=1))  ## May be reduce mean better?

        with tf.variable_scope("loss") as scope:
            self.loss = loss1 + loss2





if __name__ == '__main__':
    model = OddOneOutNet();
    sess = tf.InteractiveSession()
    now = datetime.now()
    if (file_const.tensorbaord_file == None):
        tb_path = file_const.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
    else:
        tb_path = file_const.tensorbaord_dir + file_const.tensorbaord_file
    train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    print('Saved at ',tb_path)

    train_writer.close();
