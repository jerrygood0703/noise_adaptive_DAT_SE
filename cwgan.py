from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from data_utils import *
import os
from tqdm import tqdm
from hyperparameters import Hyperparams as hp


class NoiseAdaptiveSE(object):
    def __init__(self, 
                g_net, c_net,
                data_sclean, data_snoisy,
                data_tclean, data_tnoisy,
                data_slabel, data_tlabel,
                log_path, model_path,
                frame_size, NOISETYPES, lamb_domain,
                model_type,
                lr=1e-4,
                ):

        self.model_path = model_path
        self.log_path = log_path
        self.NOISETYPES = NOISETYPES
        self.model_type = model_type

        self.frame_size = frame_size
        self.lamb_domain = lamb_domain
        self.lr = lr
        self.g_net = g_net
        self.c_net = c_net

        self.slabel = data_slabel
        self.tlabel = data_tlabel,
        self.source_clean = data_sclean
        self.source_noisy = data_snoisy
        self.target_clean = data_tclean
        self.target_noisy = data_tnoisy

        self.noise_type = tf.concat([data_slabel, data_tlabel], axis=0)
        self.clean = tf.concat([data_sclean, data_tclean], axis=0)
        self.noisy = tf.concat([data_snoisy, data_tnoisy], axis=0)
        self.G_summs = []

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement = True)
        self.config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=self.config)

    def loss(self):
        '''
        Original flow
        '''
        g_x, feats                  = self.g_net(self.noisy, reuse=False, is_training=True)
        source_recon, target_recon  = tf.split(g_x, 2, axis=0) ### First half is from source domain
        if self.model_type == "adap":
            c_f                     = self.c_net(feats, reuse=False, is_training=True)
        # ====================
        # Learning rate decay
        # ====================
        global_step = tf.Variable(0, trainable=False)
        learning_rate = self.lr
        domain_weight = self.lamb_domain
        # ============================
        # Building objective functions
        # ============================   
        self.loss = dict()
        ### Only use source clean for reconstruction loss!!!
        if self.model_type == "upper":
            recons_loss = tf.losses.absolute_difference(self.clean, g_x)
        else:
            recons_loss = tf.losses.absolute_difference(self.source_clean, source_recon) 
        ###
        ### Classify multiple noise types
        ###
        if self.model_type == "adap":
            domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.noise_type, logits=c_f))
            G_loss = recons_loss - (domain_weight * domain_loss)
        else:
            G_loss = recons_loss
        # ===================
        # # For summaries
        # ===================
        sum_l_G = []    
        sum_l_G.append(tf.summary.scalar('recons', recons_loss))       
        sum_l_G.append(tf.summary.image('enhanced', tf.transpose(target_recon, [0, 3, 2, 1]), max_outputs=6))
        sum_l_G.append(tf.summary.image('clean', tf.transpose(self.target_clean, [0, 3, 2, 1]), max_outputs=6))
        sum_l_G.append(tf.summary.scalar('lr', self.lr))
        sum_l_G.append(tf.summary.scalar('G_loss', G_loss))
        if self.model_type == "adap":
            sum_l_G.append(tf.summary.scalar('domain', domain_loss))
            sum_l_G.append(tf.summary.scalar('domain_weight', domain_weight))
            domain_acc = tf.keras.metrics.categorical_accuracy(
                            tf.one_hot(self.noise_type, self.NOISETYPES), 
                            c_f)
            sum_l_G.append(tf.summary.scalar('domain_acc', tf.reduce_mean(domain_acc)))

        self.G_summs = [sum_l_G]
        ### Count number of trainable variables
        print('%s parameters:%d' % (self.g_net.name, np.sum([np.prod(v.get_shape().as_list()) for v in self.g_net.vars])))
        print('%s parameters:%d' % (self.c_net.name, np.sum([np.prod(v.get_shape().as_list()) for v in self.c_net.vars])))

        g_opt = None
        c_opt = None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
                    .minimize(G_loss, var_list=self.g_net.vars, global_step=global_step)
            if self.model_type == "adap":                   
                c_opt = tf.train.AdamOptimizer(learning_rate=learning_rate*5, beta1=0.5, beta2=0.9)\
                        .minimize(domain_loss, var_list=self.c_net.vars)

        return g_opt, c_opt

    def train(self, mode="train", iters=65000, latest_step=None):
        g_opt, c_opt = self.loss()

        if tf.gfile.Exists(self.log_path+"G"):
            tf.gfile.DeleteRecursively(self.log_path+"G")
        tf.gfile.MkDir(self.log_path+"G")

        g_merged = tf.summary.merge(self.G_summs)
        G_writer = tf.summary.FileWriter(self.log_path+"G", self.sess.graph)

        self.sess.run(tf.global_variables_initializer()) 
        save_path = self.model_path
        print('Training...')
        #-----------------------------------------------------------------#
        saver = tf.train.Saver(max_to_keep=100)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        try:
            while not coord.should_stop():                
                for i in tqdm(range(iters)):
                    fetch_list = []
                    if self.model_type == "adap":
                        fetch_list += [c_opt]
                
                    if i%100==0:
                        fetched = self.sess.run([g_merged])  
                        G_writer.add_summary(fetched[-1], i)
                    else:
                        _ = self.sess.run(fetch_list + [g_opt])  

                    if i % hp.SAVEITER == 0:
                        saver.save(self.sess, save_path + 'model', global_step=i)

                    if i == iters-1:
                        saver.save(self.sess, save_path + 'model', global_step=i+1)
                        coord.request_stop()

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit')
        finally:
            coord.request_stop()
        coord.join(threads)

        return

    def test(self, test_list=None, specific_iter=None, mode='test'):
        FRAMELENGTH = self.frame_size
        OVERLAP = self.frame_size
        test_path = self.model_path

        if mode == 'test':
            reuse = False
            if specific_iter == 0:
                start_iter = hp.SAVEITER
                end_iter = hp.SAVEITER*50
            else:
                start_iter = specific_iter
                end_iter = specific_iter
            print('Testing...')

        elif mode == 'valid':
            reuse = True
            start_iter = specific_iter
            end_iter = specific_iter

        inputs      = tf.placeholder("float", [None, 1, hp.f_bin, self.frame_size], name='input_noisy')
        enhanced, feats = self.g_net(inputs, reuse=reuse, is_training=False)

        _pathname = os.path.join(test_path, "enhanced")
        check_dir(_pathname)       
        nlist = [x[:-1] for x in open(test_list,'r').readlines()]
        for latest_step in range(start_iter, end_iter+hp.SAVEITER, hp.SAVEITER):
            ### Load model          
            if mode == 'test':
                saver = tf.train.Saver()
                saver.restore(self.sess, test_path + "model-" + str(latest_step))
                print(latest_step)

            ### Enhanced waves in the nlist
            enh_list = []
            mse_list = []
            for name in nlist:
                if name == '':
                    continue
                _pathname = os.path.join(test_path,"enhanced",str(latest_step))
                check_dir(_pathname)
                ### Read file                   
                spec, phase, x = make_spectrum(name, is_slice=False, feature_type=hp.feature_type, mode=hp.nfeature_mode)                    
                spec_length = spec.shape[1]

                ''' run sliced spectrogram '''
                ## Pad spectrogram
                temp = np.zeros((spec.shape[0], ((spec_length-FRAMELENGTH)//OVERLAP+1)*OVERLAP+FRAMELENGTH))
                temp[:,:spec_length] = spec 

                ### Slice spectrogram into segments
                slices = []
                for i in range(0, temp.shape[1]-FRAMELENGTH+1, OVERLAP):
                    slices.append(temp[:,i:i+FRAMELENGTH])
                slices = np.array(slices).reshape((-1, 1, hp.f_bin, self.frame_size))

                ### Run graph
                spec_temp = np.zeros(temp.shape)
                output = self.sess.run(enhanced, {inputs:slices})
                for i,out_frame in enumerate(output):
                    spec_temp[:,i*OVERLAP:i*OVERLAP+FRAMELENGTH] += out_frame[0,:,:]
                spec_temp = spec_temp[:,:spec_length] 

                recons_y = recons_spec_phase(spec_temp, phase, feature_type=hp.feature_type) 
                y_out = librosa.util.fix_length(recons_y, x.shape[0])

                temp_name = name.split('/')
                _pathname = os.path.join(_pathname,temp_name[-4])
                check_dir(_pathname)                
                _pathname = os.path.join(_pathname,temp_name[-3])
                check_dir(_pathname)
                _pathname = os.path.join(_pathname,temp_name[-2])
                check_dir(_pathname)
                _pathname = os.path.join(_pathname,temp_name[-1])
                wav.write(_pathname, hp.SR, np.int16(y_out*32767))
        return 
