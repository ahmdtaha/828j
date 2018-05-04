import argparse
from datetime import datetime
from nets.two_stream import TwoStreamNet
import configuration as file_const
import constants as const
import os
import sys
import tensorflow as tf
import tf_tuple_generator as tuple_gen
import utils


def train_two_stream(dataset_path, input_train_list_files,
                     input_val_list_files, activities_path, batch_size,
                     supervision_mode):

    train_set = tuple_gen.build_input(dataset_path, input_train_list_files,
                                      activities_path, batch_size,
                                      supervision_mode, 'train')
    val_set = tuple_gen.build_input(dataset_path, input_val_list_files,
                                    activities_path, batch_size, 'val')

    train_iter = train_set.make_initializable_iterator()
    center_frames_op, motion_encodings_op, class_labels_op, filenames_op = \
        train_iter.get_next()

    val_iter = val_set.make_initializable_iterator()
    val_center_frames_op, val_motion_encodings_op, val_class_labels_op, _ = \
        val_iter.get_next()

    save_model_dir = file_const.model_save_path

    load_alex_weights = True

    if supervision_mode == 'supervised':
        img2vec_model = TwoStreamNet(
            supervised=True, load_alex_weights=load_alex_weights,
            train_spatial_tower=False, train_motion_tower=True,
            input_words=center_frames_op, input_context=motion_encodings_op,
            class_labels=class_labels_op)

        model_loss = img2vec_model.supervised_loss
        model_accuracy = img2vec_model.supervised_accuracy
    elif supervision_mode == 'unsupervised':
        img2vec_model = TwoStreamNet(
            load_alex_weights=load_alex_weights, input_words=center_frames_op,
            input_context=motion_encodings_op, class_labels=class_labels_op)

        model_loss = img2vec_model.loss
        model_accuracy = img2vec_model.accuracy
    else:
        sys.exit('Invalid supervision mode %s' % supervision_mode)

    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)

    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)

    for v in tf.trainable_variables():
        print(v.name, '\t', v.shape)

    # sess = tf.InteractiveSession()
    sess = tf.Session()
    now = datetime.now()
    if file_const.tensorbaord_file is None:
        tb_path = file_const.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
    else:
        tb_path = file_const.tensorbaord_dir + file_const.tensorbaord_file

    print(tb_path)
    if os.path.exists(tb_path):
        latest_filepath = utils.get_latest_file(tb_path)
        print(latest_filepath)
        tb_iter = tf.train.summary_iterator(latest_filepath)
        for e in tb_iter:
            last_step = e.step

        print('Continue on previous TB file ', tb_path, ' with starting step',
              last_step)
    else:
        print('New TB file *********** ', tb_path)
        last_step = 0

    train_writer = tf.summary.FileWriter(tb_path, sess.graph)

    saver = tf.train.Saver()  # saves variables learned during training
    # tf.global_variables_initializer().run()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ckpt_file = os.path.join(save_model_dir, file_const.model_save_name)
    print('Model Path ', ckpt_file)
    if os.path.exists(save_model_dir) and len(os.listdir(save_model_dir)) > 1:
        # Try to restore everything if possible
        if supervision_mode == 'unsupervised':
            saver.restore(sess, ckpt_file)
            print('Previous Model loaded ')
        else:  # supervised
            try:
                # Try to restore everything if possible
                saver.restore(sess, ckpt_file)
                print('Model Loaded Normally')
            except:
                # If not, load as much as possible
                img2vec_model.load_pretrained(sess, ckpt_file)
                print('Pretrained Weights loaded, while some layers are'
                      'randomized')
    elif load_alex_weights:
        print('Loading img2vec_model.assign_operations:',
              len(img2vec_model.assign_operations))
        sess.run(img2vec_model.assign_operations)

    train_loss = tf.summary.scalar('Train Loss', model_loss)
    val_loss = tf.summary.scalar('Val Loss', model_loss)

    model_acc_op = tf.summary.scalar('Val Accuracy', model_accuracy)

    # if supervision_mode == 'unsupervised':
    #     pos_acc_op = tf.summary.scalar('PosAccuracy', model_accuracy)
    #     neg_acc_op = tf.summary.scalar('NegAccuracy', model_accuracy)

    sess.run(val_iter.initializer)

    epochs = 0
    step = last_step
    while True:  # indefinite number of epochs
        sess.run(train_iter.initializer)
        # run epoch
        while True:
            try:
                model_loss_value, accuracy_value, _ = sess.run([
                    model_loss, model_accuracy, train_op])

                if step % const.logging_threshold == 0:
                    print('i= ', step, ' Loss= ', model_loss_value,
                          ', Acc= %2f' % accuracy_value, ' Epoch = %2f' % ((
                           step * const.batch_size) / (file_const.epoch_size)))
                    if(step != 0):
                        # run_options = tf.RunOptions(
                        #     trace_level=tf.RunOptions.FULL_TRACE)
                        tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        # train_loss_op, _ = sess.run([train_loss, train_op])
                        train_loss_op = sess.run(train_loss)

                        # TODO: add validation here!
                        # feed_dict = gen_feed_dict(
                        #     img2vec_model, img_generator, const.Subset.VAL,
                        #     None, args)
                        # val_loss_op,accuracy_op= sess.run(
                        #     [val_loss,model_acc_op], feed_dict=feed_dict)

                        train_writer.add_run_metadata(run_metadata,
                                                      'step%03d' % step)

                        train_writer.add_summary(train_loss_op, step)
                        # train_writer.add_summary(val_loss_op, step)

                        # train_writer.add_summary(accuracy_op, step)
                        train_writer.flush()

                        if(step % 100 == 0):
                            saver.save(sess, ckpt_file)
                step = step + 1
            except tf.errors.OutOfRangeError:
                epochs += 1
                break

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', help='Path to dataset of pickels'
                        'where each video is represented by a pickle file, ')
    parser.add_argument('--input_train_list_files', help='List of input files,'
                        ' where each file is a list of videos', nargs='*')
    parser.add_argument('--input_val_list_files', help='List of input files,'
                        ' where each file is a list of videos', nargs='*')
    parser.add_argument('--activities_file_path', help='A file listing all '
                        'class labels (i.e activities)')
    parser.add_argument('--supervision_mode',
                        help='Either {supervised, unsupervised}')

    flags = parser.parse_args()

    train_two_stream(flags.dataset_path, flags.input_train_list_files,
                     flags.input_val_list_files, flags.activities_file_path,
                     const.batch_size, flags.supervision_mode)
