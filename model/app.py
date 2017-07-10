import tensorflow as tf
from tensorflow.python.client import timeline

import data
import model.network


FLAGS = tf.app.flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def train_routine(network):

    train_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).\
        minimize(network.loss, global_step=tf.Variable(0, name='global_step', trainable=False))

    queue_op = data.data_queue(
        exec_mode=FLAGS.exec_mode,
        batch_size=FLAGS.batch_size,
        epoch_num=FLAGS.epoch_num)

    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            images, labels = sess.run(queue_op)
            feed_dict = {network.image: images, network.label: labels}

            current_step = sess.graph.get_tensor_by_name('global_step')
            if current_step and current_step % FLAGS.log_interval == 0:

                run_meta = tf.RunMetadata()
                run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                sess.run(train_op, feed_dict, options=run_opt, run_metadata=run_meta)

                trace = timeline.Timeline(step_stats=run_meta.step_stats)
                train_writer = tf.summary.FileWriter('/train', sess.graph)
                train_writer.add_run_metadata(run_meta, 'step%03d' % i)
                train_writer.add_summary(None, 0)

                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())
            else:
                sess.run(train_op, feed_dict)


        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)

        while not coord.should_stop():
            pass





def test():

    test_op = None
    init_op = tf.initialize_all_variables()

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        correct_prediction = tf.equal(tf.argmax(None, 1), tf.argmax(None, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={}))
    pass


def main(_):

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)


