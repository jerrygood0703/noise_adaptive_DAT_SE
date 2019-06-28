class Hyperparams:
    SAVEITER = 100
    #### Signal Processing ####
    n_fft = 512
    hop_length = 256
    n_mels = 80 # Number of Mel banks to generate
    f_bin = n_fft//2 + 1
    SR = 16000
    feature_type  = 'lps' # logmag lps
    nfeature_mode = None # mean_std minmax
    cfeature_mode = None  