import pandas as pd
import numpy as np
import uproot
import os
from config import *
from utils import *


data_kind = 'background' #Only for backrdound, don't need to merge signal

def main():

    if data_kind not in ['signal', 'background']:
        raise Exception('i can t infer the data to process, please define the data_kind: background or sig')

    if data_kind == 'background':
        print('"\x1b[31m\"merging the files "\x1b[0m"')

        if os.path.exists(numpy_bkg):
            print('folder already existing...saving {}'.format(data_kind))
        else:
            try:
                os.makedirs(numpy_bkg)
            except:
                print('failing to create output folder')
        concatenate_file(splitted_numpy_bkg, numpy_bkg, data_kind)

    elif data_kind == 'signal':
        print('"\x1b[31m\"merging the files "\x1b[0m"')

        if os.path.exists(numpy_sig):
            print('folder already existing...saving {}'.format(data_kind))
        else:
            try:
                os.makedirs(numpy_sig)
            except:
                print('failing to create output folder')

        concatenate_file(splitted_numpy_sig, numpy_sig, data_kind)


if __name__ == "__main__":

    main()
