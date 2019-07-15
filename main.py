
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, os

from data_utils import *
from model import *
from cwgan import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.set_random_seed(1234)

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam optimizer [1e-4]")
flags.DEFINE_float("lamb_domain", 0.05, "Lamda of L_DAT [0.05]")
flags.DEFINE_integer("batch_size", 16, "Batch size [16]")
flags.DEFINE_integer("iters", 100000, "Total number of training iterations [100k]")
flags.DEFINE_integer("frame_size", 32, "Frame length [32]")
flags.DEFINE_integer("NOISETYPES", 6, "Number of noise types for D to predict [6]")
flags.DEFINE_integer("target_id", 5, "Target domain noise class (0-NOISETYPES) [5]")
flags.DEFINE_string("model_type", "adap", "Which model to train, [base/upper/adap]")
flags.DEFINE_string("mode", "train", "Training or testing [train/test]")
flags.DEFINE_string("ckpt_name", "20190325_temp/", "Checkpoint name.")
flags.DEFINE_string("root_path", "/mnt/md1/user_sylar/TIMIT_noise_adap/", "Path for directories.")
flags.DEFINE_string("train_clean_path", "/mnt/md1/user_sylar/TIMIT_SE/Clean/Train", "Path for train clean data.")
flags.DEFINE_string("train_noisy_list", "/mnt/md1/user_sylar/TIMIT_noise_adap/TrNoisy_feat/tr_cafe_220", "Path for train noisy data.")
flags.DEFINE_string("test_noisy_list", "/mnt/md1/user_sylar/TIMIT_noise_adap/TsNoisy/ts_noisy", "Path for test noisy data.")
flags.DEFINE_string("test_iter", "100000", "Specific ckpt iteration for testing [100000]")
flags.DEFINE_boolean("write_tfrecord", False, "True for making tfrecord, False for nothing [true, false]")
FLAGS = flags.FLAGS

def main(_):

    log_path = os.path.join(FLAGS.root_path,'logs',FLAGS.ckpt_name)
    model_path = os.path.join(FLAGS.root_path,'models',FLAGS.ckpt_name)
    test_list_ts = FLAGS.test_noisy_list

    record_path = FLAGS.root_path
    record_name = "/target_dataset"
    source_record_name = "/source_dataset"

    print("Mode type:%s lamb_domain:%f log:%s" % (FLAGS.model_type, FLAGS.lamb_domain, log_path))
    print("record set:%s" % (record_name))

    G=LSTM_SE(FLAGS.frame_size)
    C=LSTM_Cls(FLAGS.frame_size, FLAGS.NOISETYPES)

    check_dir(log_path)
    check_dir(model_path)

    reader = dataPreprocessor(record_path, record_name, source_record_name,
                            target_id=FLAGS.target_id,
                            noisy_list=FLAGS.train_noisy_list,
                            clean_path=FLAGS.train_clean_path,
                            frame_size=FLAGS.frame_size, shift=FLAGS.frame_size)
    if FLAGS.write_tfrecord:
        print("Writing tfrecord...")
        reader.write_tfrecord()
    s_clean,s_noisy,t_clean,t_noisy,s_label,t_label = reader.read_and_decode(batch_size=FLAGS.batch_size,num_threads=32)

    gan = NoiseAdaptiveSE(
                            G,C,
                            s_clean,s_noisy,
                            t_clean,t_noisy,
                            s_label,t_label,
                            log_path,
                            model_path,
                            FLAGS.frame_size,
                            FLAGS.NOISETYPES,
                            FLAGS.lamb_domain,
                            FLAGS.model_type,
                            FLAGS.learning_rate,
                         )

    if FLAGS.mode == 'test':
        gan.test(test_list=FLAGS.test_noisy_list,  
                 specific_iter=int(FLAGS.test_iter), 
                 mode=FLAGS.mode)
    elif FLAGS.mode == 'train':
        gan.train(FLAGS.mode, FLAGS.iters)

if __name__ == '__main__':
    tf.app.run()