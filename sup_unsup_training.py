import tensorflow as tf
from datetime import datetime
import os
import sys
import data_sampling.data_args as data_args
from nets.two_loss_two_stream import TwoLossTwoStreamNet
from nets.motion_tower import MotionTower
import constants as const
import configuration as config
import numpy as np
from pydoc import locate
from utils import os_utils
from utils.logger import root_logger as logger

def gen_feed_dict(model,data_generator,subset,supervised=True):
    words, context, lbl = data_generator.next(subset, supervised=supervised)
    if(supervised):
        unsup_lbls = np.random.randint(0,10,size=(const.batch_size,config.unsupervised_num_classes),dtype=np.int32)
        sup_lbls = lbl
        labels_balance = np.ones(const.batch_size)
    else:
        unsup_lbls = lbl
        sup_lbls = np.random.randint(0,10,(const.batch_size, config.num_classes),dtype=np.int32)
        labels_balance = np.zeros(const.batch_size)
    feed_dict = {model.input_words: words, model.input_context:context , model.supervised_labels: sup_lbls ,
                 model.unsupervised_labels: unsup_lbls,model.labels_balance:labels_balance }
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
        img2vec_model = TwoLossTwoStreamNet(mode = tf.estimator.ModeKeys.TRAIN,load_alex_weights=load_alex_weights,train_spatial_tower = False,train_motion_tower = True)
    else:
        load_alex_weights = False; ## There is no spatial tower
        img2vec_model = MotionTower(mode = tf.estimator.ModeKeys.TRAIN,train_motion_tower=True,supervised=True)

    model_loss = img2vec_model.loss




    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)


    for v in tf.global_variables():
        logger.info('Global_variables ' + str(v.name) + '\t' + str(v.shape))

    logger.info('=========================================================')
    for v in tf.trainable_variables():
        print(v.name, '\t', v.shape)
        logger.info('trainable_variables ' + str(v.name) + '\t' + str(v.shape))


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
        logger.info('Continue on previous TB file '+ tb_path+ ' with starting step'+ str(last_step))
    else:
        print('New TB file *********** ', tb_path);
        logger.info('New TB file *********** '+ tb_path)
        last_step = 0;


    train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()  # saves variables learned during training


    ckpt_file = os.path.join(save_model_dir, config.model_save_name)
    print('Model Path ',ckpt_file )
    if (os.path.exists(save_model_dir) and len(os.listdir(save_model_dir)) > 2):
        try:
            # Try to restore everything if possible
            saver.restore(sess, ckpt_file)
            print('Model Loaded Normally');
            logger.info('Model Loaded Normally')
        except:
            ## If not, load as much as possible
            img2vec_model.load_pretrained(sess, ckpt_file);
            print('Pretrained Weights loaded, while some layers are randomized')
            logger.info('Pretrained Weights loaded, while some layers are randomized')
    elif load_alex_weights:
        print('Loading img2vec_model.assign_operations:',len(img2vec_model.assign_operations));
        logger.info('Loading img2vec_model.assign_operations:'+ str(len(img2vec_model.assign_operations)))
        sess.run(img2vec_model.assign_operations);

    sup_model_loss = img2vec_model.supervised_loss
    unsup_model_loss = img2vec_model.unsupervised_loss


    sup_train_loss = tf.summary.scalar('Train_Loss', sup_model_loss )
    unsup_train_loss = tf.summary.scalar('Unsup_Train_Loss', unsup_model_loss)

    sup_val_loss = tf.summary.scalar('Val_Loss', sup_model_loss )
    unsup_val_loss = tf.summary.scalar('Unsup_Val_Loss', unsup_model_loss)

    sup_model_accuracy = img2vec_model.supervised_accuracy
    unsup_model_accuracy = img2vec_model.unsupervised_accuracy

    sup_model_acc_op = tf.summary.scalar('Val_Accuracy', sup_model_accuracy)
    unsup_model_acc_op = tf.summary.scalar('Unsup_Val_Accuracy', unsup_model_accuracy)

    logger.info('Training started')
    for step in range(last_step,const.train_iters):

        ## SUP
        feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, supervised=True);
        model_loss_value, sup_model_loss_value,sup_accuracy_value, _ = sess.run([model_loss,sup_model_loss ,img2vec_model.supervised_accuracy,train_op], feed_dict)
        #print('Sup ** Model Loss',model_loss_value,' Loss ',sup_model_loss_value,' Acc ',sup_accuracy_value)
        ## UNSUP
        feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, supervised=False);
        model_loss_value,unsup_model_loss_value, unsup_accuracy_value, _ = sess.run([model_loss, unsup_model_loss, img2vec_model.unsupervised_accuracy, train_op], feed_dict)
        #print('UnSup ** Model Loss',model_loss_value,' Loss ', unsup_model_loss_value, ' Acc ', unsup_accuracy_value)

        if(step % const.logging_threshold == 0):
            print('i= ', step, 'Sup Loss= ', sup_model_loss_value,'Unsup Loss= ', unsup_model_loss_value,
                  ', Sup Acc= %2f' % sup_accuracy_value,', Unsup Acc= %2f' % unsup_accuracy_value,
                  ' Epoch = %2f' % ((step * const.batch_size) / (config.epoch_size)));
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                ## SUP
                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, supervised=True);
                sup_train_loss_op, _ = sess.run([sup_train_loss, train_op], feed_dict=feed_dict)
                ## UNSUP
                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, supervised=False);
                unsup_train_loss_op,_= sess.run([unsup_train_loss,train_op],feed_dict=feed_dict)

                ## SUP
                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, supervised=True);
                sup_val_loss_op,sup_accuracy_op= sess.run([sup_val_loss,sup_model_acc_op], feed_dict=feed_dict)

                ## UNSUP
                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, supervised=False);
                unsup_val_loss_op, unsup_accuracy_op = sess.run([unsup_val_loss, unsup_model_acc_op], feed_dict=feed_dict)

                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(sup_train_loss_op, step)
                train_writer.add_summary(unsup_train_loss_op, step)
                train_writer.add_summary(sup_val_loss_op, step)
                train_writer.add_summary(unsup_val_loss_op, step)
                train_writer.add_summary(sup_accuracy_op, step)
                train_writer.add_summary(unsup_accuracy_op, step)

                train_writer.flush()

                if(step % 100 == 0):
                    saver.save(sess, ckpt_file)
                    if (step % config.checkpoint_frequency == 0):
                        ckpt = os.path.join(save_model_dir,str(step), config.model_save_name)
                        saver.save(sess, ckpt)



    sess.close()