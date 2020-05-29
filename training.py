from matplotlib import pyplot
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
from total_GAN import generate_real_samples, generate_fake_samples, define_gan
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2

datagen = ImageDataGenerator(
    
    rotation_range=20,
    vertical_flip= True,
    horizontal_flip=True
    
    )


def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1, generator= False)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i].reshape((256,256)))
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i].reshape((256,256)))
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def train(d_model, g_model, gan_model, dataset, n_epochs=50 , n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    ##Image data generator
    real_train = datagen.flow(trainA, batch_size=n_batch, seed=135)
    real_mask = datagen.flow(trainB, batch_size=n_batch, seed=135)
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    with open('log_aug__den_lr=0.002_loss_train.csv', 'w') as f, open('log_aug__den_lr=0.002_acc_train.csv','w') as f2:
        f.write('steps,d_loss1,d_loss2,g_loss\n')
        f2.write('steps,acc_real,acc_fake\n')
        for i, real_t, real_m in zip(range(n_steps), real_train,real_mask):
        #for i in range(n_steps):
            dataset= [real_t, real_m]
            #ct a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch, generator= True)
            #generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1, acc1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            #update discriminator for generated samples
            d_loss2, acc2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performane
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            f.write('%d, %.3f, %.3f, %.3f\n' % (i+1, d_loss1, d_loss2, g_loss))
            f2.write('%d, %.3f, %.3f\n' % (i+1, acc1, acc2))
            # summarize model performance
            if (i+1) % (bat_per_epo * 10) == 0:
                summarize_performance(i, g_model, [trainA, trainB])

image_shape = (256,256,3)
image_shape1 = (256,256,1)
# define the models
d_model = define_discriminator(image_shape, image_shape1)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()
#LOADING NORMAL IMAGES
#train_col= np.load('train_splitted.npy')
#train_masks= np.load('train_masks_splitted.npy')
#dataset= [train_col,train_masks]
#LOADIN AUGMENTED IMAGES
train_col= np.load('train_splitted.npy')
train_masks= np.load('train_masks_splitted.npy')
test_col= np.load('test_splitted.npy')
test_masks= np.load('test_masks_splitted.npy')
total_col= np.concatenate((train_col,test_col), axis=0)
total_mask=np.concatenate((train_masks,test_masks), axis=0)

#denoising
denoised_col= [cv2.GaussianBlur(x, (5, 5), 0) for x in total_col[:620]]
denoised_col= np.array(denoised_col)
dataset= [denoised_col,total_mask[:620]]
def load_real_samples(dataset):
	# load compressed arrays
	#data = load(filename)
	# unpack arrays
	X1, X2 = dataset
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
    
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

new_dataset= load_real_samples(dataset)

train(d_model, g_model, gan_model, new_dataset)