
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

write_tfrecord = False
use_waveform = False
batch_size = 16
learning_rate = 1e-4
iters = 100000
frame_size = 32
NOISETYPES = 6
target_id = 5
lamb_domain = 0.05
model_type = 'adap' ### base, upper, adap
mode = sys.argv[1] # train, test

ckpt_name = 'test/'
log_path = '/mnt/md1/user_sylar/TIMIT_noise_adap/logs/'+ckpt_name
model_path = "/mnt/md1/user_sylar/TIMIT_noise_adap/models/"+ckpt_name

clean_root_dev = "/mnt/md1/user_sylar/TIMIT_SE/Clean/Dev"
test_list_dev = "/mnt/md1/user_sylar/TIMIT_noise_adap/DevNoisy/dev_cafe"

clean_root_ts = "/mnt/md1/user_sylar/TIMIT_SE/Clean/Test"
test_list_ts = "/mnt/md1/user_sylar/TIMIT_noise_adap/TsNoisy/ts_noisy"

record_path = "/mnt/md1/user_sylar/TIMIT_noise_adap"
record_name = "/target_cafe220_lps_32_nonstat_train_exlude_test"
source_record_name = "/source0-4_lps_32_500x30_stat"

print("Mode type:%s lamb_domain:%f log:%s" % (model_type, lamb_domain, log_path))
print("record set:%s" % (record_name))

G=LSTM_SE(frame_size)
C=LSTM_Cls(frame_size, NOISETYPES)

check_dir(log_path)
check_dir(model_path)

reader = dataPreprocessor(record_path, record_name, source_record_name,
                        target_id=target_id,
                        noisy_list="/mnt/md1/user_sylar/TIMIT_noise_adap/TrNoisy/tr_noisy_cafe220",
                        clean_path="/mnt/md1/user_sylar/TIMIT_SE/Clean/Train",
                        frame_size=frame_size, shift=frame_size)
if write_tfrecord:
    print("Writing tfrecord...")
    reader.write_tfrecord()
s_clean,s_noisy,t_clean,t_noisy,s_label,t_label = reader.read_and_decode(batch_size=batch_size,num_threads=32)

gan = NoiseAdaptiveSE(
                        G,C,
                        s_clean,s_noisy,
                        t_clean,t_noisy,
                        s_label,t_label,
                        test_list_dev,
                        clean_root_dev,
                        log_path,
                        model_path,
                        frame_size,
                        NOISETYPES,
                        lamb_domain,
                        model_type,
                        lr=learning_rate,
                     )

if mode == 'test':
    gan.test(test_list=test_list_ts, 
             clean_root=clean_root_ts, 
             specific_iter=int(sys.argv[2]), 
             mode=mode)
elif mode == 'train':
    gan.train(mode, iters)