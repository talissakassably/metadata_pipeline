





def onset(file_path): 
    var_signal0 = []
    var_signal1 = []
    var_signal = []

    data = get_io(file_path).read(lazy=True) # load data

    for segment in data[0].segments: # if several channels in one segment
        signal = data[0].segments[5].analogsignals[0].load()
        signal_array = np.array(signal)
        signal_array_chn0 = signal_array[:, 0]
        signal_array_chn1 = signal_array[:, 1]

    for segment in data[0].segments: # if not # todo: adapt with if and if not skip what is commented 
        #signal = data[0].segments[5].analogsignals[0].load()
        #signal_array = np.array(signal)

        mean = sum(signal_array_chn1[:10])/len(signal_array_chn1[:10]) # mean resting potential

        interp = mean + 3 # upper threshold 
        interm = mean - 3 # lower threshold 

        # todo: change the name of the variables 
        var_signal= [i for i, x in enumerate(signal_array_chn1) if x > interp or x < interm] # detect a larger variation beyond +3 and -3

    if len(var_signal) > 0: # index list of all the detected variations
        var_signal0.append(var_signal[0])
        var_signal1.append(var_signal[-1]) 

    # takes the first index corresponding to the first variation detected 
    var0 = min(var_signal0)
    var1 = min(var_signal1)

    return ('onset', signal.times[var0], 'offset',signal.times[var1]) # returns the onset: the time of the first variation and the offset: the time of the last variation

