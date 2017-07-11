import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import data
from model.network import SquareNet


FLAGS = tf.app.flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def train_routine(network):

    sum_op = network.summary

    step_op = tf.Variable(0, name='global_step', trainable=False)

    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=FLAGS.learning_rate).minimize(network.loss, global_step=step_op)

    queue_op = data.data_queue(
        exec_mode=FLAGS.exec_mode,
        batch_size=FLAGS.batch_size,
        epoch_num=FLAGS.epoch_num)

    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with tf.Session(config=config) as sess, open(FLAGS.trace_file, 'w') as trace:
        sess.run(init_op)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            while not coord.should_stop():

                images, labels = sess.run(queue_op)
                feed_dict = {network.images: images, network.labels: labels}

                step = sess.run(step_op)
                if not step or step % FLAGS.log_interval:
                    summary, _ = sess.run([sum_op, train_op], feed_dict)
                    writer.add_summary(summary, global_step=step_op)
                else:

                    run_meta = tf.RunMetadata()
                    run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                    summary, _ = sess.run([sum_op, train_op], feed_dict, options=run_opt, run_metadata=run_meta)
                    writer.add_run_metadata(run_meta, 'step%03d' % step)
                    writer.add_summary(summary, global_step=step_op)

                    stats = timeline.Timeline(step_stats=run_meta.step_stats)
                    trace.write(stats.generate_chrome_trace_format())

                if step and not step % FLAGS.checkpoint_interval:
                    saver.save(sess, FLAGS.checkpoint_dir, global_step=step_op)

        except tf.errors.OutOfRangeError:
            pass

        finally:
            writer.close()
            coord.request_stop()
            coord.join(threads)

        while not coord.should_stop():
            pass


def test_routine(network):

    queue_op = data.data_queue(
        exec_mode=FLAGS.exec_mode,
        batch_size=FLAGS.batch_size,
        epoch_num=1)

    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        saver = tf.train.Saver(var_list=tf.all_variables())
        saver.restore(sess, FLAGS.checkpoint_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_cnt = 0

        (TP, TN, FP, FN) = 0, 1, 2, 3
        confuse_mat = np.zeros([4, data.NUM_CLASSES])

        try:

            while True:

                images, labels = sess.run(queue_op)
                logits = sess.run(network.logits, {network.images: images})

                inference = np.argmax(logits, axis=-1)
                assert labels.shape == inference.shape

                correct_labels = labels[labels == inference]
                error_labels = labels[labels != inference]

                total_cnt += len(labels)
                confuse_mat[TP] = 0

        except tf.errors.OutOfRangeError:
            pass

        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # network = SquareNet()
    # train_routine(network)
    a = np.array([1,2,3,4,5])
    b = np.array([1,7,3,9,0])
    print(np.where(a == b))
    print(np.where(a != b))
    print(np.where(a not in b))




