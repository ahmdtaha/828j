import os
import sys
import tensorflow as tf
from datetime import datetime
import data_sampling.data_args as data_args
from nets.dense_motion_tower import  MotionTower
import constants as const
import configuration as config
from data_sampling.honda_tuple_loader import HondaTupleLoader as TupleLoader
import numpy as np
from utils import os_utils
from utils.logger import root_logger as logger

def gen_feed_dict(model,data_generator,subset,fix,args):

    words, context, lbl = data_generator.next(subset,supervised=True)
    print(context.shape)
    feed_dict = {model.input_context:context , model.supervised_labels: lbl}

    return feed_dict;

if __name__ == '__main__':

    save_model_dir = config.model_save_path;
    os_utils.touch_dir(save_model_dir)
    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False

    img_generator = TupleLoader(args)
    img_generator.next(const.Subset.TRAIN,supervised=True)

    load_alex_weights = False;

    motion_model = MotionTower(mode=tf.estimator.ModeKeys.TRAIN, train_motion_tower = True)
    model_loss = motion_model.supervised_loss
    model_accuracy = motion_model.supervised_accuracy



    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)


    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)

    logger.info('=========================================================')
    for v in tf.global_variables():
        logger.info('Global_variables' + str(v.name) + '\t' + str(v.shape))

    logger.info('=========================================================')
    for v in tf.trainable_variables():
        print(v.name, '\t', v.shape)
        logger.info('trainable_variables' + str(v.name) + '\t' + str(v.shape))

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
    motion_model.load_weights(sess,saver,ckpt_file,save_model_dir,load_alex_weights);




    train_loss = tf.summary.scalar('Train_Loss', model_loss)
    val_loss = tf.summary.scalar('Val_Loss', model_loss)
    model_acc_op = tf.summary.scalar('Val_Accuracy', model_accuracy)



    for step in range(last_step,const.train_iters):

        feed_dict = gen_feed_dict(motion_model, img_generator, const.Subset.TRAIN, None, args);
        model_loss_value,accuracy_value, _ = sess.run([model_loss,model_accuracy,train_op], feed_dict)

        if(step % const.logging_threshold == 0):
            print('i= ', step, ' Loss= ', model_loss_value, ', Acc= %2f' % accuracy_value,
                  ' Epoch = %2f' % ((step * const.batch_size) / (config.epoch_size)));
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                feed_dict = gen_feed_dict(motion_model, img_generator, const.Subset.TRAIN, None, args);
                train_loss_op,_= sess.run([train_loss,train_op],feed_dict=feed_dict)

                feed_dict = gen_feed_dict(motion_model, img_generator, const.Subset.VAL, None, args);
                val_loss_op,accuracy_op= sess.run([val_loss,model_acc_op], feed_dict=feed_dict)


                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(train_loss_op, step)
                train_writer.add_summary(val_loss_op, step)


                train_writer.add_summary(accuracy_op, step)
                train_writer.flush()

                if(step % 100 == 0):
                    saver.save(sess, ckpt_file)




    sess.close()