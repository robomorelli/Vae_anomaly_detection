import pandas as pd
import numpy as np
import uproot
import os
import shutil
from config import *
from utils import *



def main(data_kind):

    data_folder = root_folder + root_file_name #root_folder is imported from config_script

    root = uproot.open(data_folder) #use uproot to see the tree in in the root root_file_name

    #read the names sample by sample
    if data_kind == 'background':
        train_sample_name = []
        for n in root.keys():
            train_sample_name.append(str(n).split(";")[0].split("'")[1])

    elif data_kind == 'signal':
        signal_name = []
        for n in root.keys():
            if 'nosys' in str(n).lower():
                signal_name.append(n)
        train_sample_name = [str(n).split(";")[0].split("'")[1] for n in signal_name]

    print('I find these branches {}'. format(train_sample_name))

    if data_kind == 'background':

        if os.path.exists(splitted_numpy_bkg):
            shutil.rmtree(splitted_numpy_bkg)
            os.makedirs(splitted_numpy_bkg)
        else:
            os.makedirs(splitted_numpy_bkg)

        for name in train_sample_name:
            cutflow_bkg(data_folder, name)

    elif data_kind == 'signal':

        if os.path.exists(numpy_sig):
            shutil.rmtree(numpy_sig)
            os.makedirs(numpy_sig)
        else:
            os.makedirs(numpy_sig)
        for name in train_sample_name:
            cutflow_sig(data_folder, name)


if __name__ == "__main__":

    # define here the name of the root file to open
    root_file_name = 'user.eschanet.allTrees_v2_0_2_bkg_NoSys.root'
    data_kind = 'background'

    # #decomment next line to process signal
    # root_file_name = 'user.eschanet.allTrees_v2_0_2_signal_1Lbb_skim.root'
    # data_kind = 'signal'

    if data_kind not in ['signal', 'background']:
        print('I m trying to infer the data data_kind: bkg or signal')
        if ('bkg' in root_file_name.lower()) | ('background' in root_file_name.lower()):
            data_kind = 'background'
            print('it seem to be backgroud')
        elif ('sig' in root_file_name.lower()) | ('signal' in root_file_name.lower()):
            data_kind = 'signal'
            print('it seem to be signal')
        else:
            raise Exception('i can t infer the data to process, please define the data_kind: background or sig')
    elif data_kind == 'background':
        if ('bkg' in root_file_name.lower()) | ('background' in root_file_name.lower()):
            print('data checked')
        else:
            raise Exception('inconsistent data kind ad root file to read')
    elif data_kind == 'signal':
        if ('sig' in root_file_name.lower()) | ('signal' in root_file_name.lower()):
            print('data checked')
        else:
            raise Exception('inconsistent data kind ad root file to read')

    main(data_kind)
