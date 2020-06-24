import os
import streamlit as st

import numpy as np
def file_selector(folder_path='data/mri_crop/'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.img')]

    selected_filename = st.multiselect('Select an MRI file from the options', filenames)

    if selected_filename:
        return os.path.join(folder_path, selected_filename[0])



def normalizeImageIntensityRange(img):

    max_intensity = np.max(img)
    min_intensity = np.min(img)

    return (img - min_intensity) / (max_intensity - min_intensity)



def saveSlice(img, fname, path,mask=False):


    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')
