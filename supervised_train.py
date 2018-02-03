import tensorflow as tf
from datetime import datetime
import os
import sys
import data_sampling.data_args as data_args
from nets.two_stream import TwoStreamNet
import constants as const
import file_constants as file_const
from data_sampling.tuple_loader import TupleLoader
import numpy as np


def gen_feed_dict(model,data_generator,subset,fix,args):
    if args[data_args.gen_nearby_frame]:
        words,nearby, context, lbl = data_generator.next(subset, fix,supervised=True)
        feed_dict = {model.input_words: words,model.nearby_words: nearby, model.input_context: context, model.supervised_labels: lbl}
    else:
        words, context, lbl = data_generator.next(subset,fix,supervised=True)
        feed_dict = {model.input_words: words, model.input_context: context, model.supervised_labels: lbl}

    return feed_dict;

if __name__ == '__main__':

    save_model_dir = file_const.model_save_path;

    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False

    img_generator = TupleLoader(args)
    img_generator.next(const.Subset.TRAIN)

    classification = True



    img2vec_model = TwoStreamNet(supervised=True,train_alexnet=True)
    model_loss = img2vec_model.supervised_loss
    model_accuracy = img2vec_model.supervised_accuracy



    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(model_loss)


    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)

    train_op = optimizer.apply_gradients(grads)

    variables_names = [v.name for v in tf.trainable_variables()]
    for var_name in variables_names:
        print(var_name)

    sess = tf.InteractiveSession()
    now = datetime.now()
    train_writer = tf.summary.FileWriter(file_const.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S"), sess.graph)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()  # saves variables learned during training

    #sess.run(img2vec_model.assign_operations)
    #img2vec_model.print_means(sess);
    ckpt_file = os.path.join(save_model_dir, file_const.model_save_name)
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



    train_loss = tf.summary.scalar('Train Loss', model_loss)
    val_loss = tf.summary.scalar('Val Loss', model_loss)
    model_acc_op = tf.summary.scalar('Val Accuracy', model_accuracy)



    for step in range(const.train_iters):

        feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
        model_loss_value,accuracy_value, _ = sess.run([model_loss,model_accuracy,train_op], feed_dict)


        if(step % const.logging_threshold == 0):
            print('i= ', step, ' Loss= ', model_loss_value, ', Acc= %2f' % accuracy_value);
            if(step != 0):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.TRAIN, None, args);
                train_loss_op,_= sess.run([train_loss,train_op],feed_dict=feed_dict)

                feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, None, args);
                val_loss_op,accuracy_op= sess.run([val_loss,model_acc_op], feed_dict=feed_dict)


                ## Inspect true positive (TP), FP, TN, TP per class
                f = open('validation_classes.txt', 'w')
                for class_i in range(file_const.num_classes):
                    feed_dict = gen_feed_dict(img2vec_model, img_generator, const.Subset.VAL, class_i, args);
                    prediction = sess.run(img2vec_model.class_prediction, feed_dict=feed_dict)
                    #bins = np.bincount(prediction,minlength=10);
                    bins = np.histogram(prediction, np.arange(0, file_const.num_classes+1, 1))[0]
                    for j in range(file_const.num_classes):
                        f.write(str(bins[j])+'\t')
                    f.write("\n")
                f.close()


                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                train_writer.add_summary(train_loss_op, step)
                train_writer.add_summary(val_loss_op, step)


                train_writer.add_summary(accuracy_op, step)
                train_writer.flush()

                if(step % 100 == 0):
                    saver.save(sess, ckpt_file)



    sess.close()