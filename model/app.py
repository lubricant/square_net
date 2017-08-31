import logging

import numpy as np
import tensorflow as tf
from tensorflow.python import gfile as gf
from tensorflow.python.client import timeline

import data
import model
from model.network import HCCR_GoogLeNet


FLAGS = tf.app.flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


def training_routine(network, queue_op):

    logging.info('Training procedure starting ...')

    step_op = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('Solver'):

        learn_rate = FLAGS.learning_rate
        if FLAGS.exp_decay:
            learn_rate_delta = FLAGS.learning_rate - FLAGS.final_learning_rate
            learn_rate_decay = tf.train.exponential_decay(learn_rate_delta, step_op,
                                                          FLAGS.decay_interval, FLAGS.decay_rate, staircase=True)
            learn_rate = tf.add(learn_rate_decay, FLAGS.final_learning_rate)
            tf.summary.scalar('learning_rate', learn_rate)

        assert FLAGS.model_solver in ('SGD', 'Momentum', 'AdaDelta', 'Adam')

        solver = None

        if FLAGS.model_solver == 'SGD':
            solver = tf.train.GradientDescentOptimizer(learn_rate)
        if FLAGS.model_solver == 'Momentum':
            solver = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)
        if FLAGS.model_solver == 'AdaDelta':
            solver = tf.train.AdadeltaOptimizer(learn_rate)
        if FLAGS.model_solver == 'Adam':
            solver = tf.train.AdamOptimizer(learn_rate)

        train_op = solver.minimize(network.loss, global_step=step_op)

    log_op = tf.summary.merge_all()

    with tf.name_scope('Probe'):
        c = [network.conv1, network.conv2]
        g_w1 = tf.gradients(network.conv1, [network.conv1.vars.weight]) + tf.gradients(network.loss, [network.conv1.vars.weight])
        g_w2 = tf.gradients(network.conv2, [network.conv2.vars.weight]) + tf.gradients(network.loss, [network.conv2.vars.weight])
        f = [network.fc, network.logits]
        probe_op = c + g_w1 + g_w2 + f

    with tf.name_scope('Initializer'):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        saver = tf.train.Saver(name='Saver', max_to_keep=15)
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

                mean, stddev = np.mean(images), np.std(images)
                images -= mean
                images /= stddev

                feed_dict = {network.images: images,
                             network.labels: labels,
                             network.keep_prob: FLAGS.keep_prob}

                step, _ = sess.run([step_op, train_op], feed_dict)

                if not step % FLAGS.log_interval or step == 1:

                    run_meta = tf.RunMetadata()
                    run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                    summary, probe = sess.run([log_op, probe_op], feed_dict, options=run_opt, run_metadata=run_meta)

                    writer.add_run_metadata(run_meta, 'step%03d' % step)
                    writer.add_summary(summary, global_step=step)

                    c1, c2, gw1_co, gw1_lo, gw2_co, gw2_lo, f1, f2 = probe
                    logging.info('Report step: {}'.format(step))
                    print('-' * 150)
                    print('conv1 >\t zeros: {}%\t mean: {}\t '.format(round(np.sum(c1 <= 0)/np.prod(c1.shape), 4)*100, np.mean(c1)))
                    print('conv2 >\t zeros: {}%\t mean: {}\t '.format(round(np.sum(c2 <= 0)/np.prod(c2.shape), 4)*100, np.mean(c2)))
                    print('conv/weight1 >\t mean: {}\t\t min: {}\t\t max: {}\t\t'.format(np.mean(gw1_co), np.min(gw1_co), np.max(gw1_co)))
                    print('loss/weight1 >\t mean: {}\t\t min: {}\t\t max: {}\t\t'.format(np.mean(gw1_lo), np.min(gw1_lo), np.max(gw1_lo)))
                    print('conv/weight2 >\t mean: {}\t\t min: {}\t\t max: {}\t\t'.format(np.mean(gw2_co), np.min(gw2_co), np.max(gw2_co)))
                    print('loss/weight2 >\t mean: {}\t\t min: {}\t\t max: {}\t\t'.format(np.mean(gw2_lo), np.min(gw2_lo), np.max(gw2_lo)))
                    print('fc1 >\t zeros: {}%\t mean: {}\t '.format(round(np.sum(f1 <= 0)/np.prod(f1.shape), 4)*100, np.mean(f1)))
                    print('fc2 >\t zeros: {}%\t mean: {}\t '.format(round(np.sum(f2 <= 0)/np.prod(f2.shape), 4)*100, np.mean(f2)))
                    print('-' * 150)

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
        if FLAGS.restore_version is not None and FLAGS.restore_version >= 0:
            logging.info('Restoring checkpoint [%d]', FLAGS.restore_version)
            saver.restore(sess, FLAGS.checkpoint_file + '-' + str(FLAGS.restore_version))
        else:
            logging.info('Restoring last checkpoint')
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            assert checkpoint and checkpoint.model_checkpoint_path
            saver.restore(sess, checkpoint.model_checkpoint_path)
        logging.info('Loading checkpoint done')

        if FLAGS.export_model:
            logging.info('Exporting model ...')
            model_params_list = tf.get_collection(network.__class__.__name__)
            model_params_key = [p.name for p in model_params_list]
            model_params_val = sess.run(model_params_list)
            model_params = dict(zip(model_params_key, model_params_val))
            np.save(data.TEMP_ROOT + '/tmp/model_params.npy', (model_params,))
            logging.info('Exporting model done')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        (TP, FP, FN) = 0, 1, 2
        confuse_mat = np.zeros([3, data.NUM_CLASSES])

        samples_num = 0
        total_correct, total_error = 0, 0

        logging.info('Test procedure is ready')
        try:

            while True:

                images, labels = sess.run(queue_op)

                mean, stddev = np.mean(images), np.std(images)
                images -= mean
                images /= stddev

                logits = sess.run(network.logits, {network.images: images, network.keep_prob: 1.})

                inference = np.argmax(logits, axis=-1)
                assert labels.shape == inference.shape

                for correct in np.where(labels == inference)[0]:
                    confuse_mat[TP][labels[correct]] += 1
                    total_correct += 1

                for error in np.where(labels != inference)[0]:
                    pos, neg = labels[error], inference[error]
                    confuse_mat[FP][neg] += 1
                    confuse_mat[FN][pos] += 1
                    total_error += 1

                samples_num += len(labels)

        except tf.errors.OutOfRangeError:
            pass

        finally:
            coord.request_stop()
            coord.join(threads)

        reverse_dict, _ = data.label_dict()

        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = total_correct / (total_correct + total_error)
            precision = confuse_mat[TP] / (confuse_mat[TP] + confuse_mat[FP])
            recall = confuse_mat[TP] / (confuse_mat[TP] + confuse_mat[FN])

        accuracy = 0 if not np.isfinite(accuracy) else accuracy
        precision[~np.isfinite(precision)] = 0
        recall[~np.isfinite(recall)] = 0

        top_N = 10

        arg_precision = np.argsort(precision)
        arg_recall = np.argsort(recall)

        arg_precision = arg_precision[precision[arg_precision] > 0]
        arg_recall = arg_recall[recall[arg_recall] > 0]

        best_precision, worst_precision = arg_precision[-top_N:][::-1], arg_precision[:top_N]
        best_recall, worst_recall = arg_recall[-top_N:][::-1], arg_recall[:top_N]

        def arg_fmt(arg_list, criteria):
            return '\n\t\t' + '\n\t\t'.join(['%s: %f' % (reverse_dict[arg], criteria[arg]) for arg in arg_list])

        print("""
        Test Result:
        
        Samples Num: {total_num}
        
        Accuracy: {accuracy}
        
        Best {N} Precision: {best_precision}
            
        Worst {N} Precision: {worst_precision}
            
        Best {N} Recall: {best_recall}
            
        Worst {N} Recall: {worst_recall}
        """.format(N=top_N,
                   total_num=samples_num,
                   accuracy=accuracy,
                   best_precision=arg_fmt(best_precision, precision),
                   worst_precision=arg_fmt(worst_precision, precision),
                   best_recall=arg_fmt(best_recall, recall),
                   worst_recall=arg_fmt(worst_recall, recall)))

        ch_precision = []
        for arg in arg_precision:
            ch_precision.append((reverse_dict[arg], precision[arg]))

        precision_num = {
            'n_100_95': len(precision[precision > 0.95]),
            'n_95_90': len(precision[np.where((0.9 < precision) & (precision <= 0.95))]),
            'n_90_80': len(precision[np.where((0.8 < precision) & (precision <= 0.9))]),
            'n_80_70': len(precision[np.where((0.7 < precision) & (precision <= 0.8))]),
            'n_70_60': len(precision[np.where((0.6 < precision) & (precision <= 0.7))]),
            'n_60_50': len(precision[np.where((0.5 < precision) & (precision <= 0.6))]),
            'n_50_00': len(precision[precision <= 0.5]),
        }

        precision_per = {}
        for key in precision_num.keys():
            precision_per[key.replace('n', 'p')] = precision_num[key] / len(precision) * 100

        print("""
        Precision Statistic:
        [1.00 - 0.95) : {p_100_95:.2f}% ({n_100_95})
        [0.95 - 0.90) : {p_95_90:.2f}% ({n_95_90})
        [0.90 - 0.80) : {p_90_80:.2f}% ({n_90_80})
        [0.80 - 0.70) : {p_80_70:.2f}% ({n_80_70})
        [0.70 - 0.60) : {p_70_60:.2f}% ({n_70_60})
        [0.60 - 0.50) : {p_60_50:.2f}% ({n_60_50})
        [0.50 - 0.00] : {p_50_00:.2f}% ({n_50_00})
        
        Precision Detail:
        {ch_precision}
        """.format(ch_precision=ch_precision, **precision_num, **precision_per))


if __name__ == '__main__':

    def prepare_dir():
        logging.info('Preparing Dir ...')

        if not gf.Exists(FLAGS.checkpoint_dir):
            gf.MakeDirs(FLAGS.checkpoint_dir)

        if not gf.Exists(FLAGS.log_dir):
            gf.MakeDirs(FLAGS.log_dir)

            board_port = 3322
            start_bat = '@ echo off\n'
            start_bat += 'echo ' + '\necho '.join(model.setting.replace('|', ' ').splitlines()) + '\n'
            start_bat += 'tensorboard --logdir="." --port=' + str(board_port) + '\n'
            start_bat += 'pause'
            start_bat_file = open(data.TEMP_ROOT + '/tmp/start.bat', mode='w')
            start_bat_file.write(start_bat)
            start_bat_file.close()
            logging.info('Preparing Dir done')

    net = HCCR_GoogLeNet(is_training=FLAGS.is_training)
    queue = data.data_queue(
        data_set=FLAGS.data_set,
        batch_size=FLAGS.batch_size,
        epoch_num=FLAGS.epoch_num)

    if FLAGS.is_training:
        prepare_dir()
        training_routine(net, queue)
    else:
        evaluating_routine(net, queue)



