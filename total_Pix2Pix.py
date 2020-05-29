#import cudatoolkit
import numpy as np
from numpy.random import randint
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from Pix2Pix import define_discriminator
from Pix2Pix_generator import define_generator
import tensorflow as tf



def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100]  )
	return model

image_shape = (256,256,3)
image_shape1 = (256,256,1)
# define the models
d_model = define_discriminator(image_shape, image_shape1)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()

################ DATA HANDLING ####################
train_col= np.load('train_splitted.npy')
train_masks= np.load('train_masks_splitted.npy')
dataset= [train_col,train_masks]
def generate_real_samples(dataset, n_samples, patch_shape, generator= True):
	# unpack datase
    trainA, trainB = dataset
	# choose random instances
    if not generator:
        ix = randint(0, trainA.shape[0], n_samples)
    #ix = randint(0,  n_samples)
	# retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y
	# generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [trainA, trainB], y

def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

#Train discriminator for a single iteration
# select a batch of real samples
n_batch=1
n_patch=16
#[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)


#Real training function is implemnted in training.py
def train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=1, n_patch=16):
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))





#train(d_model, g_model, gan_model, dataset)