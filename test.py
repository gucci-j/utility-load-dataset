# coding: utf-8
from utils import load_dataset
from conv_vae.conv_vae import Conv_VAE

from keras.utils import np_utils
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

if __name__ == '__main__':
    img_size = (28, 28, 1)
    batch_size = 28
    original_dim = 784
    latent_dim = 2
    intermediate_dim = 256
    epochs = 1000
    epsilon_std = 1.0

    # load dataset
    folders = [
		'category_0',
		'category_1',
		'category_2',
		'category_3',
		'category_4',
		'category_5',
		'category_6',
		'category_7',
		'category_8',
		'category_9'
	]

    (x_train, y_train), (x_test, y_test) = load_dataset(root_dir=r'/Users/gucci/Downloads/jaffe',
                                  folders=folders, test_size=0.10, img_size=img_size)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train),) + img_size)
    x_test = x_test.reshape((len(x_test),) + img_size)
    # 1-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print('>>> Train Shape:', x_train.shape)
    print('>>> Test Shape:', x_test.shape)

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
    fpath = 'conv_vae_jaffe_weights_' + str(epochs) + '.h5'
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
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=np.argmax(y_test, axis=1))
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