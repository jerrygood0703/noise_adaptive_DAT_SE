import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
import scipy.io.wavfile as wav
import librosa
import random, os
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from hyperparameters import Hyperparams as hp

def check_dir(path_name):
    if not tf.gfile.Exists(path_name):
        tf.gfile.MkDir(path_name)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def slice_pad(spec, OVERLAP, seg_size=64, pad_value=0):
    ### Pad spectrogram
    F, T = spec.shape
    temp = np.ones([F, ((T-seg_size)//OVERLAP+1)*OVERLAP+seg_size], dtype=spec.dtype) * pad_value
    temp[:,:T] = spec  
    ### Slice spectrogram into segments
    slices = []
    for i in range(0, temp.shape[1]-seg_size+1, OVERLAP):
        slices.append(temp[:,i:i+seg_size])
    slices = np.array(slices).reshape((-1, 1, F, seg_size))
    return slices

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None, SHIFT=None, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    D = librosa.stft(y,center=False, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.n_fft, window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1


    if is_slice:
        Sxx = slice_pad(Sxx, SHIFT, seg_size=FRAMELENGTH, pad_value=0)

    return Sxx, phase, y

def recons_spec_phase(Sxx_r, phase, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     center=False,
                     hop_length=hp.hop_length,
                     win_length=hp.n_fft,
                     window=scipy.signal.hamming)
    return result

class dataPreprocessor(object):
    def __init__(self, record_path, record_name, source_record_name, target_id,
                noisy_list=None, clean_path=None,
                frame_size=64, shift=64):
        self.noisy_list = noisy_list
        self.clean_path = clean_path
        self.target_id = target_id
        self.record_path = record_path
        self.record_name = record_name
        self.source_record_name = source_record_name

        self.FRAMELENGTH = frame_size
        self.SHIFT = shift

    def get_noise_dict(self, file_list):
        noise_dict = dict()
        ntype_list = []
        ntype = 0
        for n_ in (tqdm(file_list)):
            name = n_.split('/')[-3]
            ### Register new noise type in dict
            if name not in noise_dict:
                noise_dict[name] = ntype
                ntype += 1
                ntype_list.append(noise_dict[name])
            else:
                ntype_list.append(noise_dict[name])
        print("Num of noise types = %d"%len(noise_dict))
        return noise_dict, np.array(ntype_list)

    def write_tfrecord(self):
        if tf.gfile.Exists(self.record_path):
            print('Folder already exists: {}\n'.format(self.record_path))
        else:
            tf.gfile.MkDir(self.record_path)

        n_files = np.array([x[:-1] for x in open(self.noisy_list).readlines()])
        noise_dict, n_id = self.get_noise_dict(n_files)
        print(noise_dict, np.unique(n_id))

        ### Shuffle it first
        shuffle_id = np.arange(len(n_files))
        random.shuffle(shuffle_id)
        n_files = n_files[shuffle_id]
        n_id = n_id[shuffle_id]

        source_n = n_files[n_id<self.target_id]
        target_n = n_files[n_id>=self.target_id]

        source_l = n_id[n_id<self.target_id]
        target_l = n_id[n_id>=self.target_id]

        out_file = tf.python_io.TFRecordWriter(self.record_path+self.source_record_name+'.tfrecord')
        out_file_2 = tf.python_io.TFRecordWriter(self.record_path+self.record_name+'.tfrecord')

        cnt1 = 0        
        for sn_, sl_ in tqdm(zip(source_n, source_l)):
            ### use noisy filename to find clean file
            sc_ = os.path.join(self.clean_path, sn_.split('/')[-1])
            assert sn_.split('/')[-1] == sc_.split('/')[-1]
                

            s_noisy_signals,_,x = make_spectrum(sn_,
                                                is_slice=True,
                                                feature_type=hp.feature_type, 
                                                mode=hp.nfeature_mode, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)
            s_clean_signals,_,s = make_spectrum(sc_,
                                                is_slice=True, 
                                                feature_type=hp.feature_type, 
                                                mode=None, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)

            for sn_sig,sc_sig in zip(
                                     s_noisy_signals, 
                                     s_clean_signals, 
                                    ):

                sn_raw = sn_sig.tostring()
                sc_raw = sc_sig.tostring()

                cnt1 += 1
                example = tf.train.Example(features=tf.train.Features(feature={
                    'sn_raw': _bytes_feature(sn_raw),
                    'sc_raw': _bytes_feature(sc_raw),
                    'label': _int64_feature(sl_),
                    }))
                out_file.write(example.SerializeToString())

        out_file.close()  
        
        cnt2 = 0
        for tn_, tl_ in tqdm(zip(target_n, target_l)):
            ### use noisy filename to find clean file
            tc_ = os.path.join(self.clean_path, tn_.split('/')[-1])
            assert tn_.split('/')[-1] == tc_.split('/')[-1]

            t_noisy_signals,_,x = make_spectrum(tn_, 
                                                is_slice=True, 
                                                feature_type=hp.feature_type, 
                                                mode=hp.nfeature_mode, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)
            t_clean_signals,_,s = make_spectrum(tc_, 
                                                is_slice=True, 
                                                feature_type=hp.feature_type, 
                                                mode=None,
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)

            for tn_sig,tc_sig in zip(
                                     t_noisy_signals, 
                                     t_clean_signals,
                                    ):

                tn_raw = tn_sig.tostring()
                tc_raw = tc_sig.tostring()

                cnt2 += 1
                example = tf.train.Example(features=tf.train.Features(feature={
                    'tn_raw': _bytes_feature(tn_raw),
                    'tc_raw': _bytes_feature(tc_raw),
                    'label': _int64_feature(tl_),
                    }))
                out_file_2.write(example.SerializeToString())

        out_file_2.close()    
        print("stationary num_samples = %d"%cnt1)
        print("nonstationary num_samples = %d"%cnt2)

    def read_and_decode(self,batch_size=16, num_threads=16):
        filename_queue = tf.train.string_input_producer([self.record_path+self.source_record_name+'.tfrecord'])        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'sn_raw': tf.FixedLenFeature([], tf.string),
                    'sc_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })

        filename_queue_2 = tf.train.string_input_producer([self.record_path+self.record_name+'.tfrecord'])        
        reader_2 = tf.TFRecordReader()
        _, serialized_example_2 = reader_2.read(filename_queue_2)
        features_2 = tf.parse_single_example(
                serialized_example_2,
                features={
                    'tn_raw': tf.FixedLenFeature([], tf.string),
                    'tc_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })

        s_wave = tf.decode_raw(features['sc_raw'], tf.float32)
        s_wave = tf.reshape(s_wave, [1, hp.f_bin, self.FRAMELENGTH])
        s_noisy = tf.decode_raw(features['sn_raw'], tf.float32)
        s_noisy = tf.reshape(s_noisy, [1, hp.f_bin, self.FRAMELENGTH])

        t_wave = tf.decode_raw(features_2['tc_raw'], tf.float32)
        t_wave = tf.reshape(t_wave,[1, hp.f_bin, self.FRAMELENGTH])
        t_noisy = tf.decode_raw(features_2['tn_raw'], tf.float32)
        t_noisy = tf.reshape(t_noisy,[1, hp.f_bin, self.FRAMELENGTH])

        s_label = tf.cast(features['label'], tf.int64)
        t_label = tf.cast(features_2['label'], tf.int64)

        return tf.train.shuffle_batch(
                                    [s_wave, s_noisy, t_wave, t_noisy, s_label, t_label],
                                    batch_size=batch_size,
                                    num_threads=num_threads,
                                    capacity=1000 + 10 * batch_size,
                                    min_after_dequeue=1000,
                                    name='wav_and_label')
