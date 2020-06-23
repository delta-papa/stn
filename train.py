import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import cv2
from utils_new import normalizeImageIntensityRange,saveSlice
from unet import unet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from losses import dice_coefficient, dice_coefficient_loss
import argparse


def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE,SEED):
    data_gen_args = dict(rescale=1./255,
#                      featurewise_center=True,
#                      featurewise_std_normalization=True,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)

    img_generator = datagen.flow_from_directory(img_path, target_size=(40,40), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=(40,40), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return (pair for pair in zip(img_generator, msk_generator))

# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE,SEED):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)

    img_generator = datagen.flow_from_directory(img_path, target_size=(40,40),class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=(40,40), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return (pair for pair in zip(img_generator, msk_generator))

def display(display_list):
    plt.figure(figsize=(15,15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])

def show_prediction(datagen, num=1):
    for i in range(0,num):


        image,mask = next(datagen)

        pred_mask = model.predict(image)[0] > 0.5

        display([image[0], mask[0], pred_mask])
"""
pred_image1 = nib.load('data/mri_crop/BG0796.img').get_fdata()
pred_left_mask1 = nib.load('data/mask_left/BG0796.img').get_fdata()
pred_right_mask1 = nib.load('data/mask_right/BG0796.img').get_fdata()

pred_image1 = normalizeImageIntensityRange(pred_image1)
"""

def run_training(opts):

    train_dir = opts.train_dir
    train_dir_slices = os.path.join(train_dir,'slices/imgs/')
    train_dir_masks = os.path.join(train_dir,'masks/imgs/')

    val_dir = opts.val_dir
    val_dir_slices = os.path.join(val_dir,'slices/imgs/')
    val_dir_masks = os.path.join(val_dir,'masks/imgs/')

    test_dir = opts.test_dir
    test_dir_slices = os.path.join(test_dir,'slices/imgs/')
    test_dir_masks = os.path.join(test_dir,'masks/imgs/')

    IMG_SIZE = (40,40)
    BATCH_SIZE = opts.batch_size
    SEED = opts.seed
    NUM_TRAIN = 386
    NUM_TEST = 158

    EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE
    EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE

    train_generator = create_segmentation_generator_train(train_dir+'/slices/', train_dir+'/masks/', BATCH_SIZE,SEED)
    val_generator = create_segmentation_generator_test(val_dir+'/slices/', val_dir+'/masks/', BATCH_SIZE,SEED)
    test_generator = create_segmentation_generator_test(test_dir+'/slices/',test_dir+'/masks/',BATCH_SIZE,SEED)

    model = unet(4,out_channels=1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=dice_coefficient_loss, metrics=[dice_coefficient])

    #model.summary()
    history=model.fit_generator(generator=train_generator,
                        steps_per_epoch=EPOCH_STEP_TRAIN,
                        validation_data=val_generator,
                        validation_steps=EPOCH_STEP_TEST,
                       epochs=500,shuffle=True)

    model.save(f'UNET_model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./Training/', help='directory to save Training Slices')
    parser.add_argument('--val_dir', type=str, default='./Validation/', help='directory to save Validation Slices ')
    parser.add_argument('--test_dir', type=str, default='./Testing/', help='directory to save Testing Slices')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--seed', type=int, default=2, help='Seed for reproducibility')

    opts = parser.parse_args()
    print(opts)

    run_training(opts)
