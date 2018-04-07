# coding: utf-8
from keras.layers import Layer
from keras import backend as K
from keras import metrics

# loss function layer
class vae_loss(Layer):
  def __init__(self, img_size, **kwargs):
    self.is_placeholder = True
    super(vae_loss, self).__init__(**kwargs)
    self.img_size = img_size

  def vae_loss(self, x, x_decoded_mean, z_sigma, z_mean):
    # クロスエントロピー
    reconst_loss = self.img_size[0] * self.img_size[1] * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean)) 
    # 事前分布と事後分布のD_KLの値
    kl_loss = - 0.5 * K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1)
    return K.mean(reconst_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    z_sigma = inputs[2]
    z_mean = inputs[3]
    loss = self.vae_loss(x, x_decoded_mean, z_sigma, z_mean)
    self.add_loss(loss, inputs=inputs)
    return x