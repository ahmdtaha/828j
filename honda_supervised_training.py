import tensorflow as tf
from datetime import datetime
import os
import sys
import data_sampling.data_args as data_args
from nets.two_stream import TwoStreamNet
import constants as const
import configuration as file_const
from data_sampling.tuple_loader import TupleLoader
import numpy as np
import utils
import data_sampling.honda_tuple_generator as honda_gen
import data_sampling.preprocess as preprocess
from sklearn.metrics import confusion_matrix
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gen_feed_dict(model,data_generator,sess,data_generator_reset):
    #print('Going to get new feeddict')
    try:
        data  = sess.run(data_generator)
    except:
        print('Need to reset data generator')
        sess.run(data_generator_reset)
        data = sess.run(data_generator)
    #print('Pre processing data')
    words , context,lbl= preprocess.data_prepare(data);
    #print('Pre processing done')
    # print('words.shape ',words.shape)
    # print('context.shape ', context.shape)
    # print('lbl.shape ', lbl.shape)
    feed_dict = {model.input_words: words, model.input_context: context, model.supervised_labels: lbl}
    #print('feeddict constructed')
    return feed_dict;

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':

    save_model_dir = file_const.model_save_path;

    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False

    with open(file_const.honda_session_path+'/train_session.txt', 'r') as fin:
        train_sessions = fin.read().strip().split('\n')
    with open(file_const.honda_session_path+'/val_session.txt', 'r') as fin:
        val_sessions = fin.read().strip().split('\n')

    train_data = honda_gen.tuple_generator(supervised=True, sessions=train_sessions,batch_size=const.batch_size)
    #    train_data = tuple_generator(supervised=False, sessions=train_sessions)
    train_iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                     train_data.output_shapes)
    next_train = train_iterator.get_next()
    train_init_op = train_iterator.make_initializer(train_data)

    val_data = honda_gen.tuple_generator(supervised=True, sessions=val_sessions, shuffled=False,batch_size=const.batch_size)
    #    val_data = tuple_generator(supervised=False, sessions=val_sessions, shuffled=False)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
    next_val = val_iterator.get_next()
    val_init_op = val_iterator.make_initializer(val_data)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(train_init_op)



    load_alex_weights = True;

    img2vec_model = TwoStreamNet(supervised=True,load_alex_weights=load_alex_weights,train_spatial_tower = False,train_motion_tower = True)

    #feed_dict = gen_feed_dict(img2vec_model, next_train, sess,train_init_op);

    model_loss = img2vec_model.supervised_loss
    model_accuracy = img2vec_model.supervised_accuracy



    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)


    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)

    for v in tf.trainable_variables():
        print(v.name , '\t',v.shape)

    sess = tf.InteractiveSession()
    now = datetime.now()
    if(file_const.tensorbaord_file == None):
        tb_path = file_const.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
    else:
        tb_path = file_const.tensorbaord_dir + file_const.tensorbaord_file

    print(tb_path)
    if (os.path.exists(tb_path)):
        latest_filepath = utils.get_latest_file(tb_path)
        print(latest_filepath)
        tb_iter = tf.train.summary_iterator(latest_filepath)
        for e in tb_iter:
            last_step = e.step;
        print('Continue on previous TB file ', tb_path, ' with starting step', last_step);
    else:
        print('New TB file *********** ', tb_path);
        last_step = 0;


    train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()  # saves variables learned during training

    #sess.run(img2vec_model.assign_operations)
    #img2vec_model.print_means(sess);
    ckpt_file = os.path.join(save_model_dir, file_const.model_save_name)
    cnf_file = os.path.join(save_model_dir, 'cnf_matrix.png');
    print('Model Path ',ckpt_file )
    if (os.path.exists(save_model_dir) and len(os.listdir(save_model_dir)) > 1):
        try:
            # Try to restore everything if possible
            saver.restore(sess, ckpt_file)
            print('Model Loaded Normally');
        except:
            ## If not, load as much as possible
            img2vec_model.load_pretrained(sess, ckpt_file);
            print('Pretrained Weights loaded, while some layers are randomized')

    elif load_alex_weights:
        print('Loading img2vec_model.assign_operations:',len(img2vec_model.assign_operations));
        sess.run(img2vec_model.assign_operations);


    train_loss = tf.summary.scalar('Train Loss', model_loss)
    val_loss = tf.summary.scalar('Val Loss', model_loss)
    model_acc_op = tf.summary.scalar('Val Accuracy', model_accuracy)


    sess.run(train_init_op)
    sess.run(val_init_op)
    confusion_matrix_gt_labels = np.array([]);
    confusion_matrix_predictions = np.array([]);
    for step in range(last_step,const.train_iters):

        feed_dict = gen_feed_dict(img2vec_model, next_train,sess,train_init_op);
        model_loss_value,accuracy_value, _ = sess.run([model_loss,model_accuracy,train_op], feed_dict)

        if(step % const.logging_threshold == 0):
            print('i= ', step, ' Loss= ', model_loss_value, ', Acc= %2f' % accuracy_value);
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()


                feed_dict = gen_feed_dict(img2vec_model, next_train,sess,train_init_op);
                train_loss_op,_= sess.run([train_loss,train_op],feed_dict=feed_dict)


                feed_dict = gen_feed_dict(img2vec_model, next_val,sess,val_init_op);
                val_loss_op,accuracy_op= sess.run([val_loss,model_acc_op], feed_dict=feed_dict)

                predictions = sess.run(img2vec_model.class_prediction, feed_dict)
                gt_labels = np.argmax(feed_dict[img2vec_model.supervised_labels], axis=1)

                confusion_matrix_gt_labels = np.append(confusion_matrix_gt_labels, gt_labels)
                confusion_matrix_predictions = np.append(confusion_matrix_predictions, predictions)




                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(train_loss_op, step)
                train_writer.add_summary(val_loss_op, step)


                train_writer.add_summary(accuracy_op, step)
                train_writer.flush()

                if(step % 100 == 0):

                    saver.save(sess, ckpt_file)

                    confusion_matrix_labels = ["Background", "Intersection passing", "railroad passing", 'Right turn',
                                               'left turn', 'intersection passing', 'left lane change',
                                               'right lane change', 'right lane branch', 'left lane branch', 'Merge',
                                               'crosswalk passing', 'U-turn']

                    cnf_matrix = confusion_matrix(confusion_matrix_gt_labels, confusion_matrix_predictions,
                                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    #print(confusion_matrix)

                    # Plot non-normalized confusion matrix
                    plt.figure()
                    plot_confusion_matrix(cnf_matrix, classes=confusion_matrix_labels,
                                          title='Confusion matrix, without normalization')

                    plt.show()
                    plt.savefig(cnf_file,dpi=300);
                    if(step % 300 ==0):
                        confusion_matrix_gt_labels = np.array([]);
                        confusion_matrix_predictions = np.array([]);






    sess.close()