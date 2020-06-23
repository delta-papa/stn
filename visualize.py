
import numpy as np
from skimage import io
import plotly.graph_objects as go
import nibabel as nib
import streamlit as st


#volume = nib.load('data/mri_crop/BG0796.img').get_fdata()

def visualize(volume):
        
    volume = np.rot90(volume[:,30:70,:])
    st.title("Upload + Classification Example")

    st.write('First image')
    r, c = volume[0].shape

    n_slices = volume.shape[0]

    height = (volume.shape[0]-1) / 10
    grid = np.linspace(0, height, n_slices)
    slice_step = grid[1] - grid[0]

    initial_slice = go.Surface(
                         z=height*np.ones((r,c)),
                         surfacecolor=np.flipud(volume[-1]),
                         showscale=False,colorscale='Gray')

    frames = [go.Frame(data=[dict(type='surface',
                              z=(height-k*slice_step)*np.ones((r,c)),
                              surfacecolor=np.flipud(volume[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, n_slices)]


    sliders = [dict(steps = [dict(method= 'animate',
                                  args= [[f'frame{k+1}'],
                                  dict(mode= 'immediate',
                                       frame= dict(duration=40, redraw= True),
                                       transition=dict(duration= 0))
                                     ],
                                  label=f'{k+1}'
                                 ) for k in range(n_slices)],
                    active=17,
                    transition= dict(duration= 0 ),
                    x=0, # slider starting position
                    y=0,
                    currentvalue=dict(font=dict(size=12),
                                      prefix='slice: ',
                                      visible=True,
                                      xanchor= 'center'
                                     ),
                   len=1.0) #slider length
               ]


    layout3d = dict(title_text='Coronal View', title_x=0.5,
                    width=600,
                    height=600,
                    scene_zaxis_range= [-0.1, 6.8],
                    sliders=sliders,
                )
    fig = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
    #fig.show()
    st.plotly_chart(fig)
