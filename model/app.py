import tensorflow as tf
from tensorflow.python.client import timeline


sys_param = tf.app.flags.FLAGS
MAX_EPOCH = sys_param.max_epoch


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1




def train():

    train_op = None
    init_op = tf.initialize_all_variables()

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        if True:
            run_meta = tf.RunMetadata()
            run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            sess.run(train_op, feed_dict={}, options=run_opt, run_metadata=run_meta)
            trace = timeline.Timeline(step_stats=run_meta.step_stats)

            train_writer = tf.summary.FileWriter('/train', sess.graph)
            train_writer.add_run_metadata(run_meta, 'step%03d' % i)
            train_writer.add_summary(None, 0)

            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        else:
            sess.run(train_op, feed_dict={})

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
    train()
