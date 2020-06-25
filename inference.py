import streamlit as st
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from utils.utils_new import file_selector, normalizeImageIntensityRange
import tensorflow as tf
from utils.visualize import visualize
import tempfile
import plotly.graph_objects as go

model = keras.models.load_model('weights/UNET_MRI_June22_val_dice_coeff_0.6.h5')

#input_buffer = st.file_uploader("Upload IMG File", type="img", encoding=None)
st.title('A Streamlit App for volumetric segmentation of the STN')
filename = file_selector() #put in utils

if filename:
    st.write('You selected `%s`' %filename)

    pred_image1 = nib.load(filename).get_fdata()

    pred_image1 = normalizeImageIntensityRange(pred_image1)


    plt.figure(figsize=(20,20))

    my_bar = st.progress(0)
    new_out = np.zeros((120,120,120))
    for z in range(25,45):

        for h in range(0,90,20):

            for w in range(0,90,20):

                my_bar.progress((z-25)/19)
                sub_image = pred_image1[h:h+40,w:w+40,z].reshape(-1,40,40,1)

                sub_image = normalizeImageIntensityRange(sub_image)

                prediction = model.predict(sub_image)[0]>0.6

                if(np.sum(prediction[:,:,0]) > 30):

                    new_out[h:h+40,w:w+40,z]+= prediction[:,:,0]


    #print("Total slices is :",i)

    new = normalizeImageIntensityRange(new_out)
    visualize(1.8*new+pred_image1)
