import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from models import Generator, Discriminator
from IPython import display

import utils  # local import
from models import Generator, Discriminator  # local import

EPOCHS = 100
LAMBDA = 100
BATCH_SIZE = 8
IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
SAVE_PATH = 'weights'
DATASET = 'cityscapes'
ff_dim = 32
num_heads = 2
patch_size = 8
embed_dim = 64
projection_dim = 64
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
num_patches = (IMG_HEIGHT // patch_size) ** 2

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

available_datasets = [
    'cityscapes',
    'edges2handbags',
    'edges2shoes',
    'facades',
    'maps',
    'night2day'
]

if DATASET not in available_datasets:
    print(f'[ERROR] dataset: {DATASET}')
    print('[INFO] please us on of the following datasets')
    for dataset in available_datasets:
        print(f'    -> {dataset}')

    exit(1)


assert IMG_WIDTH == IMG_HEIGHT, 'width and height must have same size'
_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{DATASET}.tar.gz'
device = '/device:GPU:0' if utils.check_cuda else '/cpu:0'

path_to_zip = tf.keras.utils.get_file(f'{DATASET}.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), f'{DATASET}/')


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


tf.config.run_functions_eagerly(False)

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


try:
    test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
except:
    test_dataset = train_dataset


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = Generator(input_shape, patch_size, num_patches, projection_dim, num_heads, ff_dim)
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

tf.config.run_functions_eagerly(True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.show()


def generate_batch_images(model, test_input, tar):
    for i in range(len(test_input)):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[i], tar[i], prediction[i]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


def train_step(input_image, target):
    with tf.device(device):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)


        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))


def fit(train_ds, epochs):
    for epoch in range(epochs):
        print(f'Epoch: [{epoch}/{epochs}]')

        for n, (input_image, target) in train_ds.enumerate():
            train_step(input_image, target)


    generator.save_weights(f'{SAVE_PATH}/{DATASET}-gen-weights.h5')
    discriminator.save_weights(f'{SAVE_PATH}/{DATASET}-disc-weights.h5')


if __name__ == '__main__':
    fit(train_dataset, EPOCHS)
