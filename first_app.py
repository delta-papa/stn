import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
import cv2
import time


import streamlit as st
from PIL import Image
from classify import predict

import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)



#uploaded_file = st.file_uploader("Choose an image...", type="png")
"""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_file)
    st.write('%s (%.2f%%)' % (label[1], label[2]*100))
"""


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


@st.cache
def generate_data():
    """Let's put the data in cache so it doesn't reload each time we rerun the script when modifying the slider"""
    # prepare some coordinates
    x, y, z = np.indices((6, 6, 6))

    # draw cuboids in the top left and bottom right corners, and a link between them
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

    # combine the objects into a single boolean array
    voxels = cube1 | cube2 | link

    colors = np.empty(voxels.shape, dtype=object)

    print(colors.shape)
    colors[link] = 'red'
    colors[cube1] = 'blue'
    colors[cube2] = 'green'

    return voxels, colors

voxels, colors = generate_data()

# let's put sliders to modify view init, each time you move that the script is rerun, but voxels are not regenerated
# TODO : not sure that's the most optimized way to rotate axis but well, demo purpose
azim = st.sidebar.slider("azim", 0, 90, 30, 1)
elev = st.sidebar.slider("elev", 0, 360, 240, 1)

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')
ax.view_init(azim, elev)

st.pyplot()
