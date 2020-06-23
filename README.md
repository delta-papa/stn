# 3D Segmentation of the Subthalamic Nucleus in the Brain

This repo contains code for implementing 3D Volumetric Segmentation of the Subthalamic Nucleus (STN) in the brain using a 2D U-Net. The STN is a region in the brain that is stimulated by neurosurgeons to reduce the effect of tremors in Parkinson's Disease Patients. However, the location, size and orientation of the STN is 
not fixed for each patient. Moreover, traditional techniques often make use of a Brain ATLAS to locate such regions in the brain. 

This project is a step forward to detect and segment the STN independent of an ATLAS. The STN consists of a Left and Right portion known as the Left STN (LSTN) and Right STN (RSTN) respectively. To trace each STN manually, a human annotator takes about 30 minutes. This is an arduous task for any surgeon. My model performs this segmentation in the STN.

### Dependencies
This code depends on the following libraries:

- Python >= 3.6
- Tensorflow 2.2.0 
- nibabel
- numpy = 1.16
- matplotlib = 3.2.2


## Preparing your data
- To use your own data, you will have to specify the path to the folder containing this data (--root_dir).
- Images have to be in nifti (.nii) format
- You have to split your data into two folders: Training/Validation. Each folder will contain 2 sub-folders: 1 subfolder that will contain the image modality and GT, which contain the nifti files for the images and their corresponding ground truths. 
- In the runTraining function, you have to change the name of the subfolders to the names you have in your dataset (lines 129-130 and 143-144).


### Training

The model can be trained using below command:  
```
python train.py 
```

## Current version
- The current version includes LiviaNET. We are working on including some extensions we made for different challenges (e.g., semiDenseNet on iSEG and ENIGMA MICCAI Challenges (2nd place in both))
- A version of SemiDenseNet for single modality segmentation has been added. You can choose the network you want to use with the argument --network
```
--network liviaNet  o  --network SemiDenseNet
```
- Patch size, and sampling steps values are hard-coded. We will work on a generalization of this, allowing the user to decide the input patch size and the frequence to sample the patches.
- TO-DO: 
-- Include data augmentation step.
-- Add a function to generate a mask (ROI) so that 1) isolated areas outside the brain can be removed and 2) sampling strategy can be improved. So far, it uniformly samples patches across the whole volume. If a mask or ROI is given, sampling will focus only on those regions inside the mask.

If you use this code in your research, please consider citing the following paper:

- Dolz, Jose, Christian Desrosiers, and Ismail Ben Ayed. "3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study." NeuroImage 170 (2018): 456-470.

If in addition you use the semiDenseNet architecture, please consider citing these two papers:

- [1] Dolz J, Desrosiers C, Wang L, Yuan J, Shen D, Ayed IB. Deep CNN ensembles and suggestive annotations for infant brain MRI segmentation. Computerized Medical Imaging and Graphics. 2019 Nov 15:101660.

- [2] Carass A, Cuzzocreo JL, Han S, Hernandez-Castillo CR, Rasser PE, Ganz M, Beliveau V, Dolz J, Ayed IB, Desrosiers C, Thyreau B. Comparing fully automated state-of-the-art cerebellum parcellation from magnetic resonance images. NeuroImage. 2018 Dec 1;183:150-72.

### Design of the semiDenseNet architecture
![model](images/semiDenseNet.png)

# LiviaNet_pytorch
