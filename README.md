# 3D Segmentation of the Subthalamic Nucleus in the Brain

This repo contains code for implementing 3D Volumetric Segmentation of the Subthalamic Nucleus (STN) in the brain using a 2D U-Net. The STN is a region in the brain that is stimulated by neurosurgeons to reduce the effect of tremors in Parkinson's Disease Patients. However, the location, size and orientation of the STN is 
not fixed for each patient. Moreover, traditional techniques often make use of a Brain ATLAS to locate such regions in the brain. 

This project is a step forward to detect and segment the STN independent of an ATLAS. The STN consists of a Left and Right portion known as the Left STN (LSTN) and Right STN (RSTN) respectively. To trace each STN manually, a human annotator takes about 30 minutes. This is an arduous task for any surgeon. My model performs this segmentation in the STN.

## Dependencies
This code depends on the following libraries:
```
Keras==2.4.2
tensorflow==2.2.0
Keras-Preprocessing==1.1.2

matplotlib==3.2.2
nibabel==2.5.2
nilearn==0.5.2
numpy==1.16.0
opencv-python==4.2.0.34
scikit-image==0.17.2
scikit-learn==0.20.4
streamlit==0.62.0
```
You may run the following command to install the dependencies:
```
pip install -r requirements.txt 
```

## Preparing the data
- The original data is stored in the form of .IMG and .HDR files in the ./data directory. The ./data directory has 3 subdirectories - mri_crop, mask_left, mask_right. The mri_crop directory has the MRI image files for each anonymised patient. The mask_left and mask_right have the respective Left STN and Right STN traces for the patients. For example, if a patient has an anonymous ID 'BG0844' then the patient's MRI scan would be stored in 'mri_crop' with the files 'BG0844.img' and 'BG0844.hdr'. The Left and Right STN masks of the patient will also be stored with the same filenames in the respective folders. 

Note: The reason to use the name 'mri_crop' for the directory containing the MRI images is that these are cropped scans of shape 120x120x120. The original scans themselves are of dimensions 512x512x400 but I have cropped them to the region only where the STN is present. This helps to save computational costs and also focus attention of the computer vision algorithms to the regions near the STN.

You can start by creating the training data for the 2D U-Net model by running the following command:

```
python create_dataset.py --n_train=35 --n_val=9 
```
n_train and n_val stands for the number of images you want for training and validation respectively. This script would create the training and validation slices for the Left and Right STN and store them in the Training and Validation folders. 

The Training and Valdation Folders are organized as follows:
```
|-Training
  |-masks
    |-left_stn
    |-right_stn
  |-slices
    |-left_stn
    |-right_stn
    
|-Validation
  |-masks
    |-left_stn
    |-right_stn
  |-slices
    |-left_stn
    |-right_stn
 ```
This shows that for each folder, there exists sub-directories for the human annotated masks called 'masks' and corresponding MRI 2D slices called 'slices'. Furthermore, each of those sub-directories have a left_stn and right_stn sub-directory.

## Model

I have developed a customizable 2D U-Net model that takes in a single 40x40 MRI image slice and then segments the STN from that slice. The model is defined in the file unet.py. 

## Training

The model can be trained using below command:  
```
python train.py --n_levels = 2  --batch_size=4 --seed=2 --epochs=100 --lr=1e-4 --loss=BCE
```
n_levels is the number of convolutional and max pooling blocks used in the encoder-decoder network of the U-Net. The batch size can be specified as per your choice while seed is used for reproducibility of code. Finally you can set the number of epochs, learning rate and loss as hyper-parameters for training the model. Loss can be one of BCE or DICE which stands for Binary Cross Entropy loss and Dice Loss respectively.

This script will make use of the training data created in the previous step to train the 2D U-Net model and save it in the Models directory as .h5 file. 


## Inference

To run the inference on a testing image, run the following command:

```
streamlit run inference.py

```
This launches a streamlit app that will open a new browser window where you can see the image you specified as being segmented with the STN. 
