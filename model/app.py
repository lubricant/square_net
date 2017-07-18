import logging

import numpy as np
import tensorflow as tf
from tensorflow.python import gfile as gf
from tensorflow.python.client import timeline


import data
from model.network import HCCR_GoogLeNet


FLAGS = tf.app.flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def training_routine(network, queue_op):

    logging.info('Training procedure starting ...')

    log_op = tf.summary.merge_all()

    step_op = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('Solver'):
        train_op = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate if not FLAGS.exp_decay else (
                tf.train.exponential_decay(FLAGS.learning_rate, global_step=step_op,
                                           decay_steps=FLAGS.decay_interval,
                                           decay_rate=FLAGS.decay_rate)
            )).minimize(network.loss, global_step=step_op)

    with tf.name_scope('Initializer'):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(config=config) as sess, open(FLAGS.trace_file, 'w') as tracer:
        sess.run(init_op)

        saver = tf.train.Saver(name='Saver')
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        logging.info('Loading checkpoint ...')
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            logging.info('Loading checkpoint done')
        else:
            logging.info('Loading checkpoint skipped')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logging.info('Training procedure is ready')

        try:

            while not coord.should_stop():

                images, labels = sess.run(queue_op)
                feed_dict = {network.images: images,
                             network.labels: labels,
                             network.keep_prob: FLAGS.keep_prob}

                step = sess.run(step_op) + 1
                if step % FLAGS.log_interval:
                    sess.run(train_op, feed_dict)
                else:

                    run_meta = tf.RunMetadata()
                    run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                    summary, _ = sess.run([log_op, train_op], feed_dict, options=run_opt, run_metadata=run_meta)
                    writer.add_run_metadata(run_meta, 'step%03d' % step)
                    writer.add_summary(summary, global_step=step)

                    stats = timeline.Timeline(step_stats=run_meta.step_stats)
                    tracer.write(stats.generate_chrome_trace_format())

                    logging.info('Report step: {}'.format(step))

                if not step % FLAGS.checkpoint_interval:
                    logging.info('Saving checkpoint {} ...'.format(step//FLAGS.checkpoint_interval))
                    saver.save(sess, FLAGS.checkpoint_file, global_step=step_op)
                    logging.info('Saving checkpoint done')

        except tf.errors.OutOfRangeError:
            pass

        finally:
            writer.close()
            coord.request_stop()
            coord.join(threads)

        logging.info('Training procedure is finish')
        logging.info('Total step: {}'.format(sess.run(step_op)))


def evaluating_routine(network, queue_op):

    logging.info('Test procedure starting ...')

    with tf.name_scope('Initializer'):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        saver = tf.train.Saver()

        logging.info('Loading checkpoint ...')
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        assert checkpoint and checkpoint.model_checkpoint_path
        saver.restore(sess, checkpoint.model_checkpoint_path)
        logging.info('Loading checkpoint done')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        (TP, FP, FN) = 0, 1, 2
        confuse_mat = np.zeros([3, data.NUM_CLASSES])

        total_correct, total_error = 0, 0

        logging.info('Test procedure is ready')
        try:

            while True:

                images, labels = sess.run(queue_op)
                logits = sess.run(network.logits, {network.images: images, network.keep_prob: 1.})

                inference = np.argmax(logits, axis=-1)
                assert labels.shape == inference.shape

                for correct in np.where(labels == inference):
                    confuse_mat[TP][labels[correct]] += 1
                    total_correct += 1

                for error in np.where(labels != inference):
                    pos, neg = labels[error], inference[error]
                    confuse_mat[FP][neg] += 1
                    confuse_mat[FN][pos] += 1
                    total_error += 1

        except tf.errors.OutOfRangeError:
            pass

        finally:
            coord.request_stop()
            coord.join(threads)

        reverse_dict, = data.label_dict()

        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = total_correct / (total_correct + total_error)
            precision = confuse_mat[TP] / (confuse_mat[TP] + confuse_mat[FP])
            recall = confuse_mat[TP] / (confuse_mat[TP] + confuse_mat[FN])

            accuracy = 0 if np.isfinite(accuracy) else accuracy
            precision[~np.isfinite(precision)] = 0
            recall[~np.isfinite(recall)] = 0

        top_N = 10

        arg_precision = np.argsort(precision)
        arg_recall = np.argsort(recall)

        arg_precision = arg_precision[precision[arg_precision] > 0]
        arg_recall = arg_recall[recall[arg_recall] > 0]

        best_precision, worst_precision = arg_precision[-top_N:], arg_precision[:top_N]
        best_recall, worst_recall = arg_recall[-top_N:], arg_recall[:top_N]

        def arg_fmt(arg_list):
            '\n\t\t'.join(['%s: %f' % (reverse_dict[arg], precision[arg]) for arg in arg_list])

        print("""
        Test Result:
        
        Accuracy: {accuracy}
        
        Best {N} Precision: {best_precision}
            
        Worst {N} Precision: {worst_precision}
            
        Best {N} Recall: {best_recall}
            
        Worst {N} Recall: {worst_recall}
            
        """.format(N=top_N, accuracy=accuracy,
                   best_precision=arg_fmt(best_precision),
                   worst_precision=arg_fmt(worst_precision),
                   best_recall=arg_fmt(best_recall),
                   worst_recall=arg_fmt(worst_recall)))


if __name__ == '__main__':

    def prepare_dir():
        logging.info('Preparing Dir ...')

        if not gf.Exists(FLAGS.checkpoint_dir):
            gf.MakeDirs(FLAGS.checkpoint_dir)

        if not gf.Exists(FLAGS.log_dir):
            gf.MakeDirs(FLAGS.log_dir)

        logging.info('Preparing Dir done')

    net = HCCR_GoogLeNet()
    queue = data.data_queue(
        data_set=FLAGS.data_set,
        batch_size=FLAGS.batch_size,
        epoch_num=FLAGS.epoch_num)

    if FLAGS.is_training:
        prepare_dir()
        training_routine(net, queue)
    else:
        evaluating_routine(net, queue)



