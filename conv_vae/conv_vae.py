# coding: utf-8
# dependencies:
# OS: macOS 10.13.3
# Python: 3.6.2
# Module: keras, tensorflow, numpy, matplotlib, h5py, scipy.stats

from keras.layers import Dense, Flatten, Input, Reshape, Lambda, Layer
from keras.layers import Conv2D, Deconv2D, Activation, UpSampling2D
from keras.datasets import fashion_mnist
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras import Model, metrics
from keras import backend as K
from conv_vae.vae_loss import vae_loss
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# coefficients 
# 事前係数
img_size = (28, 28, 1)
batch_size = 28
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 5
epsilon_std = 1.0

class Conv_VAE(object):
    # save coefficients in advance
    # コンストラクタで定数を先に渡しておく
    def __init__(self, img_size, original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std):
        self.img_size = img_size
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std

    def vae_model(self):
        # encoder
        x = Input(shape=img_size)
        hidden = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')(x)
        hidden = Conv2D(filters=4, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(hidden)
        hidden = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(hidden)
        hidden = Flatten()(hidden)
        hidden = Dense(self.intermediate_dim, activation='relu')(hidden)
        z_mean = Dense(self.latent_dim)(hidden)
        z_sigma = Dense(self.latent_dim)(hidden)

        # decoder
        # reparameterization trick
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_sigma])
        dense_1 = Dense(self.intermediate_dim, activation='relu')(z)
        dense_2 = Dense(7 * 7 * 16, activation='relu')(dense_1)
        reshape = Reshape((7, 7, 16))(dense_2)
        deconv_1 = Deconv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(reshape)
        upsamp_1 = UpSampling2D((2, 2))(deconv_1)
        deconv_2 = Deconv2D(filters=4, kernel_size=(3,3), padding='same', activation='relu')(upsamp_1)
        upsamp_2 = UpSampling2D((2, 2))(deconv_2)
        x_decoded_mean = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid')(upsamp_2)
        # loss function layer
        y = vae_loss(img_size=img_size)([x, x_decoded_mean, z_sigma, z_mean])

        return Model(x, y), Model(x, z_mean)
    
    # サンプル生成用デコーダ
    def generator(self, _model):
        _, dense_1, dense_2, reshape, deconv_1, upsamp_1, deconv_2, upsamp_2, x_decoded_mean, _ = _model.layers[8:]
        
        decoder_input = Input(shape=(self.latent_dim,))
        _dense_1 = dense_1(decoder_input)
        _dense_2 = dense_2(_dense_1)
        _reshape = reshape(_dense_2)
        _deconv_1 = deconv_1(_reshape)
        _upsamp_1 = upsamp_1(_deconv_1)
        _deconv_2 = deconv_2(_upsamp_1)
        _upsamp_2 = upsamp_2(_deconv_2)
        _x_decoded_mean = x_decoded_mean(_upsamp_2)

        return Model(decoder_input, _x_decoded_mean)

    def sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
        return z_mean + K.exp(z_sigma / 2) * epsilon

    def model_compile(self, model):
        model.compile(optimizer='rmsprop', loss=None)

if __name__ == '__main__':
    # Fashion-MNISTのデータセットを呼び出し
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train),) + img_size)
    x_test = x_test.reshape((len(x_test),) + img_size)
    # 1-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print('x_test shape: {0}'.format(x_test.shape))
    print('x_train shape: {0}'.format(x_train.shape))

    # generate an instance for the VAE model
    # VAEクラスからインスタンスを生成
    _vae = Conv_VAE(img_size, original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std)
    # build -> compile -> summary -> fit
    _model, _encoder = _vae.vae_model()
    _vae.model_compile(_model)
    _model.summary()

    # save history to CSV
    callbacks = []
    callbacks.append(CSVLogger("history.csv"))

    _hist = _model.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        callbacks=callbacks)

    # save weights
    fpath = 'conv_vae_mnist_weights_' + str(epochs) + '.h5'
    _model.save_weights(fpath)
    
    # plot loss
    loss = _hist.history['loss']
    val_loss = _hist.history['val_loss']
    plt.plot(range(1, epochs), loss[1:], marker='.', label='loss')
    plt.plot(range(1, epochs), val_loss[1:], marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # show q(z|x) ~ p(z)
    x_test_encoded = _encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    # 散布図を描画するメソッド: scatter(データx, y, 色c)
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c = np.argmax(y_test, axis=1))
    plt.colorbar()
    plt.show()

    # show p(x|z)
    n = 15  
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    _generator = _vae.generator(_model)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = _generator.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure,cmap='gray')
    plt.show()
    