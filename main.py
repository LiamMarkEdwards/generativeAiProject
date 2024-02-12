import numpy as np
import tensorflow as tf
from h5py._hl import dataset
from keras.src.layers.preprocessing.benchmarks.image_preproc_benchmark import BATCH_SIZE
from numpy.random import seed
from tensorflow.keras import layers, models
from IPython import display
import matplotlib.pyplot as plt


# define the genrative model here
def make_generative_model():
    model = models.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(784, activation='tanh'))
    return model


# define the discriminator model here
def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(784,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    # define the generator and the discriminator here


generator = make_generative_model()
discriminator = make_discriminator_model()

# define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def the optimiser
generator_optimiser = tf.keras.optimizers.Adam(1e-4)
discriminator_optimiser = tf.keras.optimizers.Adam(1e-4)


# Define the loss functions
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# def the training procedure for the AI gen
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimiser.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimiser.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# train the model
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 100

for epoch in range(EPOCHS):
    for image_batch in dataset:
        train_step(image_batch)

        # Generate a GIF of the training process
        # Generate and display images
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
