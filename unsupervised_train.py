import  tensorflow as tf
from datetime import datetime
import os
import sys
import data_sampling.data_args as data_args
from nets.two_stream import TwoStreamNet
import constants as const
import configuration as file_const
from data_sampling.tuple_loader import  TupleLoader
import utils


def gen_feed_dict(model,data_generator,subset,fix,args):
    if args[data_args.gen_nearby_frame]:
        words,nearby, context, lbl = data_generator.next(subset, fix)
        feed_dict = {model.input_words: words,model.nearby_words: nearby, model.input_context: context, model.unsupervised_labels: lbl}
    else:
        words, context, lbl = data_generator.next(subset,fix)
        feed_dict = {model.input_words: words, model.input_context: context, model.unsupervised_labels: lbl}

    return feed_dict;


if __name__ == '__main__':
    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False


    img_generator = TupleLoader(args)
    img_generator.next(const.Subset.TRAIN)

    load_alex_weights = True;
    img2vec_model = TwoStreamNet(load_alex_weights=load_alex_weights)
    model_loss = img2vec_model.loss
    model_accuracy = img2vec_model.accuracy



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
    if (file_const.tensorbaord_file == None):
        tb_path = file_const.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
    else:
        tb_path = file_const.tensorbaord_dir + file_const.tensorbaord_file

    print(tb_path )
    if(os.path.exists(tb_path )):
        latest_filepath = utils.get_latest_file(tb_path)
        print(latest_filepath)
        tb_iter = tf.train.summary_iterator(latest_filepath)
        for e in tb_iter:
            last_step = e.step;
        print('Continue on previous TB file ',tb_path,' with starting step',last_step);
    else:
        print('New TB file *********** ',tb_path);
        last_step = 0;
    train_writer = tf.summary.FileWriter(tb_path , sess.graph)

    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()

    ckpt_file = os.path.join(file_const.model_save_path, file_const.model_save_name)
    if (os.path.exists(file_const.model_save_path) and len(os.listdir(file_const.model_save_path)) > 1):
        saver.restore(sess, ckpt_file)
        print('Previous Model loaded ')
    elif load_alex_weights:
        print('Loading img2vec_model.assign_operations:', len(img2vec_model.assign_operations));
        sess.run(img2vec_model.assign_operations);



    train_loss = tf.summary.scalar('Train Loss', model_loss)
    val_loss = tf.summary.scalar('Val Loss', model_loss)


    model_acc_op = tf.summary.scalar('Val Accuracy', model_accuracy)
    pos_acc_op = tf.summary.scalar('PosAccuracy', model_accuracy)
    neg_acc_op = tf.summary.scalar('NegAccuracy', model_accuracy)



    for step in range(last_step,const.train_iters):

        feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
        model_loss_value,accuracy_value, _ = sess.run([model_loss,model_accuracy,train_op], feed_dict)

        #debug_list = [img2vec_model.diff,img2vec_model.smoothness_loss,train_op]
        #print(sess.run(debug_list, feed_dict),feed_dict)

        if(step % const.logging_threshold == 0):
            print('i= ', step, ' Loss= ', model_loss_value, ', Acc= %2f' % accuracy_value, ' Epoch = %2f' % ((step * const.batch_size)/(file_const.epoch_size)));
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
                train_loss_op,_= sess.run([train_loss,train_op],feed_dict=feed_dict)

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, None, args);
                val_loss_op,accuracy_op= sess.run([val_loss,model_acc_op], feed_dict=feed_dict)


                ## Allow me to inspect true positive and false positive rates
                # feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, 1, args);
                # pos_acc_value = sess.run(pos_acc_op, feed_dict=feed_dict)

                ## Allow me to inspect true negative and false negative rates
                # feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, -1, args);
                # neg_acc_value  = sess.run(neg_acc_op, feed_dict=feed_dict)

                #print(summary)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(train_loss_op, step)
                train_writer.add_summary(val_loss_op, step)

                # train_writer.add_summary(pos_acc_value, step)
                # train_writer.add_summary(neg_acc_value, step)

                train_writer.add_summary(accuracy_op, step)
                train_writer.flush()

                if(step % 100 == 0):
                    ckpt_file = os.path.join(file_const.model_save_path, file_const.model_save_name)
                    saver.save(sess, ckpt_file)



    sess.close()