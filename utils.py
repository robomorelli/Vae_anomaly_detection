import pandas as pd
import numpy as np
import uproot
import os
import random
from subprocess import check_output
from sklearn.model_selection import train_test_split
from config import *

def cutflow_bkg(data_folder, train_sample_name):

    entrysteps=3000000
    tot = 0

    # for n in TrainSamplesName:
    print('processing {}[{}]'.format(data_folder, train_sample_name))

    events = uproot.open(data_folder)[train_sample_name]
    name = train_sample_name.split('_')[0]
    array = events.lazyarray('met')

    print('numero di eventi:', len(array))
    file_split = len(array)//entrysteps
    start_name_file = 0
    entrystart = start_name_file*entrysteps

    batches = events.iterate(columns, entrystart=entrystart,
                           entrysteps=entrysteps,
                               outputtype=pd.DataFrame)

    for ix in range(start_name_file, file_split+1):

        print('I splitted the root file: processsing {}/{} of {}'.format(ix+1, file_split+1, name))
        batch = next(batches)
        print('adding luminosity')
        batch['luminosity'] = 139000
        print(len(batch))

        batch = batch[batch['nLep_signal'].astype(int)==1]
        print('after signal {}'.format(len(batch)))

        batch = batch[batch['trigMatch_metTrig'].astype(int)==1]
        print('after trig {}'.format(len(batch)))

        batch = batch[(batch['jet2Pt']>30) | (batch['jet3Pt']>30)]
        batch = batch[batch['jet4Pt']<30]
#         batch = batch[((batch['jet2Pt']>30)|(batch['jet3Pt']>30))]
        print('after jetpt {}'.format(len(batch)))

        batch = batch[((batch['nBJet30_MV2c10']>=1)&(batch['nBJet30_MV2c10']<4))]
#         batch = batch[batch['nBJet30_MV2c10']==2]
        print('after bjet {}'.format(len(batch)))

        batch = batch[batch['met']>220]
        print('after met {}'.format(len(batch)))

        batch = batch[batch['mt']>50]
        print('after mt {}'.format(len(batch)))

        batch = batch[((batch['mbb']>=100)&(batch['mbb']<=140))]
        print('after mbb {}'.format(len(batch)))

        batch = batch[batch['mct2']>100]
        print('after mct2 {}'.format(len(batch)))

        if len(batch) > 0:

            batch['weight'] = batch.apply(lambda row: row['genWeight']*row['eventWeight']*row['pileupWeight']*
                                 row['leptonWeight']*row['bTagWeight']*row['jvtWeight']*row['luminosity'], axis=1)

            batch_fin = batch.iloc[:,:8]
            batch_fin['weight'] = batch['weight']

            batch_fin = batch_fin[['met', 'mt', 'mbb', 'mct2',
            'mlb1','lep1Pt', 'nJet30', 'nBJet30_MV2c10', 'weight']]

            tot = tot + len(batch)
            print('tot = {}'.format(tot))
            print("\x1b[31m\"saving {}_{}""\x1b[0m".format(name,ix))

            np.save(splitted_numpy_bkg + 'background_{}_{}.npy'.format(name,ix), batch_fin.values)


    return


def cutflow_sig(data_folder, train_sample_name):

    entrysteps=3000000
    tot = 0

    # for n in TrainSamplesName:
    print('processing {}[{}]'.format(data_folder, train_sample_name))

    events = uproot.open(data_folder)[train_sample_name]
    name = '_'.join(train_sample_name.split('_')[1:-1])
    array = events.lazyarray('met')

    print('numero di eventi:', len(array))
    file_split = len(array)//entrysteps
    start_name_file = 0
    entrystart = start_name_file*entrysteps

    batches = events.iterate(columns_sig, entrystart=entrystart,
                           entrysteps=entrysteps,
                               outputtype=pd.DataFrame)

    for ix in range(start_name_file, file_split+1):

        print('I splitted the root file: processsing {}/{} of {}'.format(ix+1, file_split+1, name))
        batch = next(batches)
        print('adding luminosity')
        batch['luminosity'] = 139000
        print(len(batch))

        batch = batch[batch['nLep_signal'].astype(int)==1]
        print('after signal {}'.format(len(batch)))

        batch = batch[batch['trigMatch_metTrig'].astype(int)==1]
        print('after trig {}'.format(len(batch)))

        # batch = batch[(batch['jet2Pt']>30) | (batch['jet3Pt']>30)]
        # batch = batch[batch['jet4Pt']<30]
#         batch = batch[((batch['jet2Pt']>30)|(batch['jet3Pt']>30))]
        # print('after jetpt {}'.format(len(batch)))

        batch = batch[((batch['nBJet30_MV2c10']>=1)&(batch['nBJet30_MV2c10']<4))]
#         batch = batch[batch['nBJet30_MV2c10']==2]
        print('after bjet {}'.format(len(batch)))

        batch = batch[batch['met']>220]
        print('after met {}'.format(len(batch)))

        batch = batch[batch['mt']>50]
        print('after mt {}'.format(len(batch)))

        batch = batch[((batch['mbb']>=100)&(batch['mbb']<=140))]
        print('after mbb {}'.format(len(batch)))

        batch = batch[batch['mct2']>100]
        print('after mct2 {}'.format(len(batch)))

        if len(batch) > 0:

            batch['weight'] = batch.apply(lambda row: row['genWeight']*row['eventWeight']*row['pileupWeight']*
                                 row['leptonWeight']*row['bTagWeight']*row['jvtWeight']*row['luminosity'], axis=1)

            batch_fin = batch.iloc[:,:8]
            batch_fin['weight'] = batch['weight']

            batch_fin = batch_fin[['met', 'mt', 'mbb', 'mct2',
            'mlb1','lep1Pt', 'nJet30', 'nBJet30_MV2c10', 'weight']]

            tot = tot + len(batch)
            print('tot = {}'.format(tot))
            print("\x1b[31m\"saving {}_{}""\x1b[0m".format(name,ix))

            np.save(numpy_sig + 'signal_{}_{}.npy'.format(name,ix), batch_fin.values)

    return


def concatenate_file(input_data_folder, output_data_folder, data_kind):
    fh_name = os.listdir(input_data_folder)

    for i, name in enumerate(fh_name):
        if i == 0:
            joined_file = np.load(input_data_folder + '{}'.format(name))

        else:
            new_to_join = np.load(input_data_folder + '{}'.format(name))
            joined_file = np.concatenate((joined_file, new_to_join))

    np.save(output_data_folder + '{}.npy'.format(data_kind), joined_file)

    return


def train_val_test_split(input_data_folder, output_data_folder, proportions):

    fh_name = os.listdir(input_data_folder)
    back_tot = np.load(input_data_folder + '{}'.format(fh_name[0]))

    random.seed(30)

    y = [0 for x in range(len(back_tot))]

    if len(proportions) == 2:

        X_train, X_val, _, _ = train_test_split(back_tot, y, test_size=proportions[1], random_state=42)

        np.save(output_data_folder + '/background.npy', back_tot)
        np.save(output_data_folder + '/background_train.npy', X_train)
        np.save(output_data_folder + '/background_val.npy', X_val)

    elif len(proportions) == 3:

        X_train, X_val_test, _, _ = train_test_split(back_tot, y, test_size=proportions[1]+proportions[2], random_state=42)

        y = [0 for x in range(len(X_val_test))]

        remaining_fraction = 1 - proportions[0]
        factor = proportions[2]/remaining_fraction

        print('factor is {} and test is {}'.format(factor, remaining_fraction*factor))

        X_val, X_test, _, _ = train_test_split(X_val_test, y, test_size=factor, random_state=42)

        np.save(output_data_folder + '/background.npy', back_tot)
        np.save(output_data_folder + '/background_train.npy', X_train)
        np.save(output_data_folder + '/background_val.npy', X_val)
        np.save(output_data_folder + '/background_test.npy', X_val)

    else:
        raise Exception('error: no splitting')

    return
