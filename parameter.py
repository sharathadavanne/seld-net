def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,    # To do quick test. Trains/test on small subset of dataset
        azi_only=True,      # Estimate Azimuth only

        # Dataset loading parameters
        echoic='anechoic',  # Dataset to use: seld, bigseld, bigseldamb, echoic, anechoic, circ, circrev
        overlap=1,         # maximum number of overlapping sound events [1, 2, 3]
        split=1,           # Cross validation split [1, 2, 3]
        db=30,             # SNR of sound events.
        nfft=512,          # FFT/window length size

        # DNN Model parameters
        sequence_length=512,        # Feature sequence length
        batch_size=16,              # Batch size
        dropout_rate=0.0,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        pool_size=[8, 8, 2],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        xyz_def_zero=True,          # Use default DOA Cartesian value x,y,z = 0,0,0
        nb_epochs=1000,             # Train for maximum epochs

        # Not important
        mode='regr',        # Only regression ('regr') supported as of now
        nb_cnn3d_filt=32,   # For future. Not relevant for now
        cnn_3d=False,       # For future. Not relevant for now
        weakness=0          # For future. Not relevant for now
    )
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("Using default parameters")

    # Different datasets
    elif argv == '2':  # anechoic data set
        params['echoic'] = 'anechoic'
        params['sequence_length'] = 512

    elif argv == '3':  # echoic data set
        params['echoic'] = 'echoic'
        params['sequence_length'] = 256

    elif argv == '4':  # circ data set
        params['echoic'] = 'circ'
        params['sequence_length'] = 256

    elif argv == '5':  # circrev data set
        params['echoic'] = 'circrev'
        params['sequence_length'] = 256

    # anechoic circular array data set split 1, overlap 3
    elif argv == '6':  #
        params['echoic'] = 'circ'
        params['overlap'] = 3
        params['split'] = 1

    # anechoic Ambisonic data set with sequence length 64 and batch size 32
    elif argv == '7':  #
        params['echoic'] = 'anechoic'
        params['sequence_length'] = 64
        params['batch_size'] = 32

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("{}: {}".format(key, value))
    return params
