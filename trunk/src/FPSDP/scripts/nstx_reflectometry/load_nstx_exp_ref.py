"""Short script to load the nstx experimental reflectometry data
"""

import FPSDP.Diagnostics.Reflectometry.NSTX.nstx as nstx

data_path = '/p/gkp/lshi/NSTX_REF/raw_data/'
shot_num = '139047'
freqs = [30,32.5,35,37.5,42.5,45,47.5,50,55,57.5,60,62.5,67.5,70,72.5,75]




def make_file_name(data_path,shot_num,freq):
    """construct the output file name based on the shot number and frequency

    Arguments:
        shot_num: string,
        freq: double, in GHz
    return:
        string, 
    """

    return '{0}{1}_{2:4.3}GHz.h5'.format(data_path,shot_num,float(freq))

def get_loaders(data_path,shot_num,freqs):
    """make all the loaders ready
    return:
        list of nstx.NSTX_REF_Loader objects
    """
    loaders = []
    for freq in freqs:
        file_name = make_file_name(data_path,shot_num,freq)
        loaders.append(nstx.NSTX_REF_Loader(file_name))

    return loaders

loaders = get_loaders(data_path,shot_num,freqs)
analyser = nstx.Analyser(loaders)
