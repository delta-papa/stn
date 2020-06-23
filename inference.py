import streamlit as st
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from utils_new import file_selector, normalizeImageIntensityRange
import tensorflow as tf

model = keras.models.load_model('UNET_MRI_June22_val_dice_coeff_0.6.h5')


filename = file_selector() #put in utils

st.write('You selected `%s`' % filename)

if filename:

    pred_image1 = nib.load(filename).get_fdata()

    pred_image1 = normalizeImageIntensityRange(pred_image1)


    plt.figure(figsize=(20,20))


    new_out = np.zeros((120,120,120))
    for z in range(25,45):

        for h in range(0,90,10):

            for w in range(0,90,10):
                sub_image = pred_image1[h:h+40,w:w+40,z].reshape(-1,40,40,1)

                sub_image = normalizeImageIntensityRange(sub_image)

                prediction = model.predict(sub_image)[0]>0.8

                new_out[h:h+40,w:w+40,z]+= prediction[:,:,0]


    #print("Total slices is :",i)

    new = normalizeImageIntensityRange(new_out)

    st.image(pred_image1[:,:,35]+new[:,:,35], caption='Uploaded Image.', use_column_width=True)
