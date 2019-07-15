class Hyperparams:
    SAVEITER = 2000
    #### Signal Processing ####
    n_fft = 512
    hop_length = 256
    f_bin = n_fft//2 + 1
    SR = 16000
    feature_type  = 'lps' # logmag lps
    nfeature_mode = None # mean_std minmax
    cfeature_mode = None  