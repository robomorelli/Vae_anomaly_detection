

root_folder = 'root_data/'
splitted_numpy_bkg = 'splitted_numpy_bkg/'
splitted_numpy_sig = 'splitted_numpy_sig/'
numpy_bkg = 'numpy_data/background/'
numpy_sig = 'numpy_data/signal/'
train_val_test = 'numpy_data/train_val_test/'

model_results_single = 'model_results/single_train/'
model_results_multiple = 'model_results/multiple_train/'
plot_results = 'plot_results/'

columns = ['met', 'mt', 'mbb', 'mct2',
           'mlb1', 'nJet30', 'lep1Pt', 'nBJet30_MV2c10', 'jet1Pt',
           'trigMatch_metTrig', 'jet2Pt',
           'jet3Pt','jet4Pt', 'nLep_signal',
           'genWeight','eventWeight', 'pileupWeight',
           'leptonWeight','bTagWeight','jvtWeight']

columns_sig = ['met', 'mt', 'mbb', 'mct2',
               'mlb1', 'nJet30', 'lep1Pt', 'nBJet30_MV2c10',
               'genWeight','eventWeight', 'pileupWeight',
               'leptonWeight','bTagWeight','jvtWeight',
               'trigMatch_metTrig', 'nLep_signal']

cols = ['met', 'mt', 'mbb', 'mct2',
        'mlb1', 'lep1Pt', 'nJet30','nBJet30_MV2c10', 'weight']
