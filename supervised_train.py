import tensorflow as tf
from datetime import datetime
import os
import sys
import data_sampling.data_args as data_args
from nets.two_stream import TwoStreamNet
from nets.motion_tower import MotionTower
import constants as const
import configuration as config
import numpy as np
from pydoc import locate
from utils import os_utils

def gen_feed_dict(model,data_generator,subset,fix,args):

    words, context, lbl = data_generator.next(subset,supervised=True)
    feed_dict = {model.input_words: words, model.input_context:context , model.supervised_labels: lbl}

    return feed_dict;

if __name__ == '__main__':

    save_model_dir = config.model_save_path;
    os_utils.touch_dir(save_model_dir)
    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False

    img_generator_class = locate(config.db_tuple_loader)
    img_generator = img_generator_class(args)

    img_generator.next(const.Subset.TRAIN,supervised=True)
    if config.use_two_stream:
        load_alex_weights = True;
        img2vec_model = TwoStreamNet(supervised=True,load_alex_weights=load_alex_weights,train_spatial_tower = False,train_motion_tower = True)
    else:
        load_alex_weights = False; ## There is no spatial tower
        img2vec_model = MotionTower(mode = tf.estimator.ModeKeys.TRAIN,train_motion_tower=True,supervised=True)

    model_loss = img2vec_model.supervised_loss
    model_accuracy = img2vec_model.supervised_accuracy



    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)


    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)

    trained_variables = []
    for v in tf.trainable_variables():
        print(v.name, '\t', v.shape)
        trained_variables.append(str(v.name) + '\t' + str(v.shape))
    os_utils.txt_write(os.path.join(save_model_dir, 'trained_var.txt'), trained_variables)

    sess = tf.InteractiveSession()
    now = datetime.now()
    if(config.tensorbaord_file == None):
        tb_path = config.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
    else:
        tb_path = config.tensorbaord_dir + config.tensorbaord_file

    print(tb_path)
    if (os.path.exists(tb_path)):
        latest_filepath = os_utils.get_latest_file(tb_path)
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
    ckpt_file = os.path.join(save_model_dir, config.model_save_name)
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



    for step in range(last_step,const.train_iters):

        feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
        model_loss_value,accuracy_value, _ = sess.run([model_loss,model_accuracy,train_op], feed_dict)

        if(step % const.logging_threshold == 0):
            print('i= ', step, ' Loss= ', model_loss_value, ', Acc= %2f' % accuracy_value,
                  ' Epoch = %2f' % ((step * const.batch_size) / (config.epoch_size)));
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
                train_loss_op,_= sess.run([train_loss,train_op],feed_dict=feed_dict)

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, None, args);
                val_loss_op,accuracy_op= sess.run([val_loss,model_acc_op], feed_dict=feed_dict)


                # if(step % 1000 == 0):
                #     ## Inspect true positive (TP), FP, TN, TP per class
                #     val_acc= np.zeros((file_const.num_classes,file_const.num_classes));
                #     for class_i in range(file_const.num_classes):
                #         feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, class_i, args);
                #         prediction = sess.run(img2vec_model.class_prediction, feed_dict=feed_dict)
                #         #bins = np.bincount(prediction,minlength=10);
                #         bins = np.histogram(prediction, np.arange(0, file_const.num_classes+1, 1))[0]
                #         val_acc[class_i,:] = bins;
                #     utils.pkl_write('./dump/val_acc.pkl',val_acc);


                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(train_loss_op, step)
                train_writer.add_summary(val_loss_op, step)


                train_writer.add_summary(accuracy_op, step)
                train_writer.flush()

                if(step % 100 == 0):
                    saver.save(sess, ckpt_file)



    sess.close()