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
                feed_dict = {network.image: images, network.label: labels}

                step = sess.run(step_op)
                if not step or step % FLAGS.trace_interval:
                    sess.run(train_op, feed_dict)
                else:
                    run_meta = tf.RunMetadata()
                    run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                    summary, _ = sess.run([sum_op, train_op], feed_dict, options=run_opt, run_metadata=run_meta)

                    stats = timeline.Timeline(step_stats=run_meta.step_stats)
                    trace.write(stats.generate_chrome_trace_format())

                    writer.add_run_metadata(run_meta, 'step_%03d' % step)
                    writer.add_summary(summary, global_step=step_op)

                    saver.save(sess, FLAGS.checkpoint_dir, global_step=step_op)

        except tf.errors.OutOfRangeError:
            pass

        finally:
            coord.request_stop()
            coord.join(threads)

        while not coord.should_stop():
            pass


def test_routine():

    test_op = None
    init_op = tf.initialize_all_variables()

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        correct_prediction = tf.equal(tf.argmax(None, 1), tf.argmax(None, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={}))
    pass


if __name__ == '__main__':
    network = SquareNet()
    train_routine(network)





