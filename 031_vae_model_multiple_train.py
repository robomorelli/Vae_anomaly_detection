import numpy as np
import pandas as pd

from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras import layers as KL
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, Callback, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.constraints import max_norm
from numpy.random import seed

from sklearn.externals.joblib import dump, load

import json
import os
import shutil
import random as rn

from sklearn import preprocessing

from config import *
from vae_utility import *

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

back = np.load(train_val_test + '/background_train.npy')
back_val = np.load(train_val_test + '/background_val.npy')

train = back[:,:-1]
val = back_val[:,:-1]

original_dim = 8
Nf_lognorm = 6
Nf_PDgauss = 2

######DEFINE FUNCTIONS AND CLASS TO BUILD THE MODEL######

def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = K.tf.multiply(K.square(sigma), K.square(sigma_prior))
    kl_loss += K.square(K.tf.divide(mu_prior - mu, sigma_prior))
    kl_loss += K.log(K.tf.divide(sigma_prior, sigma)) -1
    return 0.5 * K.sum(kl_loss, axis=-1)

def RecoProb_forVAE(x, par1, par2, par3, w = 2):
    N = 0
    nll_loss = 0

    w_i = 2

    #Log-Normal distributed variables
    mu = par1[:,:w_i]
    sigma = par2[:,:w_i]
    fraction = par3[:,:w_i]
    x_clipped = K.clip(x[:,:w_i], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,:w_i], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(single_NLL, axis=-1)

    #Log-Normal distributed variables
    mu = par1[:,w_i:w_i+1]
    sigma = par2[:,w_i:w_i+1]
    fraction = par3[:,w_i:w_i+1]
    x_clipped = K.clip(x[:,w_i:w_i+1], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,w_i:w_i+1], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))

    nll_loss += K.sum(single_NLL*w, axis=-1)

    mu = par1[:,w_i+1:Nf_lognorm]
    sigma = par2[:,w_i+1:Nf_lognorm]
    fraction = par3[:,w_i+1:Nf_lognorm]
    x_clipped = K.clip(x[:,w_i+1:Nf_lognorm], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,w_i+1:Nf_lognorm], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))

    nll_loss += K.sum(single_NLL, axis=-1)

    N += Nf_lognorm

    mu = par1[:,N:N+Nf_PDgauss]
    sigma = par2[:,N:N+Nf_PDgauss]
    norm_xp = K.tf.divide(x[:,N:N+Nf_PDgauss] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,N:N+Nf_PDgauss] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(single_NLL, axis=-1)

    return nll_loss


class CustomKLLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomKLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, sigma, mu_prior, sigma_prior = inputs
        return KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior)

def IdentityLoss(y_train, NETout):
    return K.mean(NETout)

### AUTOENCODER CLASS TO SAVE AUTOENCODER###

class SaveAutoencoder(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
         save_best_only=False, save_weights_only=False, mode='auto', period=1):

        super(ModelCheckpoint, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:

            warnings.warn('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.' % (mode),
                      RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
            else:
                    self.monitor_op = np.less
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        autoencoder = Model(inputs=self.model.input,
                                 outputs=[self.model.get_layer('Output_par1').output,
                                         self.model.get_layer('Output_par2').output,
                                          self.model.get_layer('Output_par3').output])


        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            autoencoder.save_weights(filepath, overwrite=True)
                        else:
                            autoencoder.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    autoencoder.save_weights(filepath, overwrite=True)
                else:
                    autoencoder.save(filepath, overwrite=True)

def train_vae(name, w, intermediate_dim, act_fun, latent_dim, kernel_max_norm,
              lr, epochs, weight_KL_loss, batch_size, num_train):

    for num_model in range(num_train):

        class CustomRecoProbLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomRecoProbLayer, self).__init__(**kwargs)

            def call(self, inputs):
                x, par1, par2, par3 = inputs
                return RecoProb_forVAE(x, par1, par2, par3, w = w)


        ########### MODEL ###########
        intermediate_dim = intermediate_dim
        act_fun = act_fun
        latent_dim = latent_dim
        kernel_max_norm = kernel_max_norm
        lr = lr
        epochs = epochs
        weight_KL_loss = weight_KL_loss
        batch_size = batch_size

        # The below is necessary for starting Numpy generated random numbers
        # in a well-defined initial state.
        np.random.seed(42)
        # The below is necessary for starting core Python generated random numbers
        # in a well-defined state.
        rn.seed(12345)

        x_DNN_input = Input(shape=(original_dim,), name='Input')
        hidden_1 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h1')
        aux = hidden_1(x_DNN_input)

        hidden_2 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h2')

        aux = hidden_2(aux)

        L_z_mean = Dense(latent_dim, name='Latent_mean')
        T_z_mean = L_z_mean(aux)
        L_z_sigma_preActivation = Dense(latent_dim, name='Latent_sigma_h')

        aux = L_z_sigma_preActivation(aux)
        L_z_sigma = Lambda(InverseSquareRootLinearUnit, name='Latent_sigma')
        T_z_sigma = L_z_sigma(aux)

        L_z_latent = Lambda(sampling, name='Latent_sampling')([T_z_mean, T_z_sigma])
        decoder_h1 = Dense(intermediate_dim,
                           activation=act_fun,
                           kernel_constraint=max_norm(kernel_max_norm),
                           name='Decoder_h1')(L_z_latent)

        decoder_h2 = Dense(intermediate_dim, activation=act_fun, name='Decoder_h2')(decoder_h1)

        L_par1 = Dense(original_dim, name='Output_par1')(decoder_h2)

        L_par2_preActivation = Dense(original_dim , name='par2_h')(decoder_h2)
        L_par2 = Lambda(InverseSquareRootLinearUnit, name='Output_par2')(L_par2_preActivation)

        L_par3_preActivation = Dense(Nf_lognorm, name='par3_h')(decoder_h2)
        L_par3 = Lambda(ClippedTanh, name='Output_par3')(L_par3_preActivation)

        fixed_input = Lambda(SmashTo0)(x_DNN_input)
        h1_prior = Dense(1,
                         kernel_initializer='zeros',
                         bias_initializer='ones',
                         trainable=False,
                         name='h1_prior'
                        )(fixed_input)

        L_prior_mean = Dense(latent_dim,
                             kernel_initializer='zeros',
                             bias_initializer='zeros',
                             trainable=True,
                             name='L_prior_mean'
                            )(h1_prior)

        L_prior_sigma_preActivation = Dense(latent_dim,
                                            kernel_initializer='zeros',
                                            bias_initializer='ones',
                                            trainable=True,
                                            name='L_prior_sigma_preAct'
                                           )(h1_prior)
        L_prior_sigma = Lambda(InverseSquareRootLinearUnit, name='L_prior_sigma')(L_prior_sigma_preActivation)

        params = KL.concatenate([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma, L_par1, L_par2, L_par3], axis=1)

        L_RecoProb = CustomRecoProbLayer(name='RecoNLL')([x_DNN_input, L_par1, L_par2, L_par3])
        L_KLLoss = CustomKLLossLayer(name='KL')([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma])
        vae = Model(inputs=x_DNN_input, outputs=[L_KLLoss, L_RecoProb])

        adam = optimizers.adam(lr)

        def metric_wapper(layer):

            N = 4*latent_dim

            par1 = layer[:, N: N+original_dim]
            N += original_dim

            par2 = layer[:, N: N+original_dim]
            N += original_dim

            par3 = layer[:, N:N+Nf_lognorm]

            mu = layer[:, :latent_dim]
            sigma = layer[:, latent_dim: 2*latent_dim]
            mu_prior = layer[:, 2*latent_dim: 3*latent_dim]
            sigma_prior = layer[:, 3*latent_dim: 4*latent_dim]

            def metric(y_true, y_pred):

                KL = weight_KL_loss*KL_loss_forVAE(mu,sigma,mu_prior,sigma_prior)
                RecoLoss = RecoProb_forVAE(y_true[:, :original_dim], par1, par2, par3, w=1)

                return (K.mean(RecoLoss+KL))

            return metric

        metric = metric_wapper(params)

        vae.compile(optimizer=adam,
                    loss=[IdentityLoss, IdentityLoss],
                    loss_weights=[weight_KL_loss, 1.]
                    , metrics=[metric])


        train_history = vae.fit(x=train, y=[train, train],
            validation_data = (val, [val, val]),
            shuffle=True,
            epochs=epochs,

            batch_size=batch_size,
            callbacks =
                        [TerminateOnNaN(),
                        ModelCheckpoint(model_results_multiple + '{}/vae_{}_{}.h5'.format(name, name, num_model),
                                            monitor='val_loss',
                                            mode='auto', save_best_only=True,verbose=1,
                                            period=1),
                            EarlyStopping(monitor='val_loss', patience=50, verbose=1, min_delta=0.3),
                            ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.8,
                                              patience=5,
                                              mode = 'auto',
                                              epsilon=0.01,
                                              cooldown=0,
                                              min_lr=9e-8,
                                              verbose=1),
                        SaveAutoencoder(model_results_multiple + '{}/autoencoder_{}_{}.h5'.format(name, name, num_model),
                                            monitor='val_loss',
                                            mode='auto', save_best_only=True,verbose=1,
                                            period=1)
                                            ])

        hist_df = pd.DataFrame(train_history.history)

        # or save to csv:
        hist_csv_file = model_results_multiple +'{}/history_{}_{}.csv'.format(name, name, num_model)
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

if __name__ == "__main__":

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)

    name = 'vae'

    try:
        os.makedirs(model_results_multiple + name)
    except:
        pass

    train_vae(name, w=2, intermediate_dim=50, act_fun='relu',latent_dim=4, kernel_max_norm=500,
                lr=0.003,epochs=2000, weight_KL_loss=0.6, batch_size=200, num_train=15)
