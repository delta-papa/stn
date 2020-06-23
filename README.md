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


## Preparing the data
- The data is stored in the form of .IMG and .HDR files in the ./data directory. The ./data directory has 3 subdirectories - mri_crop, mask_left, mask_right. 
The mri_crop directory has the MRI image files for each anonymised patient. The mask_left and mask_right have the respective Left STN and Right STN traces for the patients. For example, if a patient has an anonymous ID 'BG0844' then the patient's MRI scan would be stored in 'mri_crop' with the files 'BG0844.img' and 'BG0844.hdr'. The Left and Right STN masks of the patient will also be stored with the same filenames in the respective folders. 

Note: The reason to use the name 'mri_crop' for the directory containing the MRI images is that these are cropped scans of shape 120x120x120. The original scans themselves are of dimensions 512x512x400 but I have cropped them to the region only where the STN is present. This helps to save computational costs and also focus attention of the computer vision algorithms to the regions near the STN.

You can start by creating the training data for the 2D U-Net model by running the following command:

```
python create_dataset.py --n_train=35 --n_val=9
```


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
