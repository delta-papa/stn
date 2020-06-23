import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import tensorflow
import nilearn
import cv2
from niwidgets import NiftiWidget
from nilearn.plotting import view_img, glass_brain, plot_anat, plot_epi
from utils_new import normalizeImageIntensityRange,saveSlice
import streamlit as st
st.title("Upload + Classification Example")


mri_path = 'data/mri_crop/'

left_stn_path = 'data/mask_left/'

right_stn_path = 'data/mask_right/'

patients = [f for f in os.listdir(mri_path) if f.endswith('.img')]

train_dir = 'Training/'

train_dir_slices = os.path.join(train_dir,'slices/imgs')
train_dir_masks = os.path.join(train_dir,'masks/imgs')

val_dir = 'Validation/'

val_dir_slices = os.path.join(val_dir,'slices/imgs/')
val_dir_masks = os.path.join(val_dir,'masks/imgs')

test_dir = 'Testing/'

test_dir_slices = os.path.join(test_dir,'slices/imgs/')
test_dir_masks = os.path.join(test_dir,'masks/imgs')

"""
def saveSlice(img, fname, path,mask=False):


    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')

"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
#                      featurewise_center=True,
#                      featurewise_std_normalization=True,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)

    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return (pair for pair in zip(img_generator, msk_generator))



# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)

    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return (pair for pair in zip(img_generator, msk_generator))

"""
IMG_SIZE = (40,40)
BATCH_SIZE = 4
SEED = 2
NUM_TRAIN = 386
NUM_TEST = 158



train_generator = create_segmentation_generator_train('Training/slices/', 'Training/masks/', BATCH_SIZE)


val_generator = create_segmentation_generator_test('Validation/slices/', 'Validation/masks/', 4)


test_generator = create_segmentation_generator_test('Testing/slices/','Testing/masks/',4)

"""

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

from tensorflow import keras
model = keras.models.load_model('UNET_MRI_June22_val_dice_coeff_0.6.h5')

def show_prediction(datagen, num=1):
    for i in range(0,num):


        image,mask = next(datagen)

        pred_mask = model.predict(image)[0] > 0.5

        display([image[0], mask[0], pred_mask])

pred_image1 = nib.load('data/mri_crop/BG0796.img').get_fdata()
pred_left_mask1 = nib.load('data/mask_left/BG0796.img').get_fdata()
pred_right_mask1 = nib.load('data/mask_right/BG0796.img').get_fdata()

pred_image1 = normalizeImageIntensityRange(pred_image1)
"""
#uploaded_file = st.file_uploader("Choose an image...", type="img")


def file_selector(folder_path='data/mri_crop/'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.img')]

    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()


st.write('You selected `%s`' % filename)

if filename:

    pred_image1 = nib.load(filename).get_fdata()

    pred_image1 = normalizeImageIntensityRange(pred_image1)


    plt.figure(figsize=(20,20))
    i=1

    new_out = np.zeros((120,120,120))
    for z in range(25,45):


        for h in range(0,90,10):

            for w in range(0,90,10):
                sub_image = pred_image1[h:h+40,w:w+40,z].reshape(-1,40,40,1)

                sub_image = normalizeImageIntensityRange(sub_image)

                prediction = model.predict(sub_image)[0]>0.8

                new_out[h:h+40,w:w+40,z]+= prediction[:,:,0]

                #plt.imshow(prediction[:,:,0],cmap='gray')
                #print(np.sum(prediction))
                #fewrf
                i+=1
        #print("Slices in one slice is ",i)
    print("Total slices is :",i)

    new = normalizeImageIntensityRange(new_out)

    st.image(pred_image1[:,:,35]+new[:,:,35], caption='Uploaded Image.', use_column_width=True)



    #plt.imshow(new[:,:,35],cmap='gray')

    #my_widget
