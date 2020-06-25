import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from utils_new import normalizeImageIntensityRange,saveSlice


def create_dataset(opts):

    training_folder = opts.train_dir
    validation_folder = opts.val_dir
    n_train = opts.n_train
    n_val = opts.n_val
    mri_path = opts.mri_data
    left_stn_path = opts.mask_left
    right_stn_path = opts.mask_right

    if n_train+n_val>44:
        return "Maximum data-set size is 45. You are asking for more"


    patients = [f for f in os.listdir(mri_path) if f.endswith('.img')]

    for r in range(n_train+n_val):
        img = nib.load(os.path.join(mri_path,patients[r])).get_fdata()
        left_mask = nib.load(os.path.join(left_stn_path,patients[r])).get_fdata()
        right_mask = nib.load(os.path.join(right_stn_path,patients[r])).get_fdata()

        a,b,c = np.where((left_mask)!=0)

        e,f,g = np.where(right_mask!=0)

        if r<n_train:
            folder = training_folder

        else:
            folder = validation_folder


        for i in range(min(c),max(c)+1):


            if(np.sum(right_mask[min(e)+abs(min(e)-max(e))//2-20:min(e)+abs(min(e)-max(e))//2+20,
                              min(f)+abs(min(f)-max(f))//2-20:min(f)+abs(min(f)-max(f))//2+20,i]))>=30:


                  n_img = normalizeImageIntensityRange(img[min(e)+abs(min(e)-max(e))//2-20:min(e)+abs(min(e)-max(e))//2+20,
                                  min(f)+abs(min(f)-max(f))//2-20:min(f)+abs(min(f)-max(f))//2+20,i])

                  #save the right stn mri slice


                  saveSlice(n_img,str(patients[r])+'_slice_'+str(i),folder+'slices/right_stn/')

                  saveSlice(right_mask[min(e)+abs(min(e)-max(e))//2-20:min(e)+abs(min(e)-max(e))//2+20,
                                 min(f)+abs(min(f)-max(f))//2-20:min(f)+abs(min(f)-max(f))//2+20,i],
                            str(patients[r])+'_slice_'+str(i),folder+'masks/right_stn/')


            if(np.sum(left_mask[min(a)+abs(min(a)-max(a))//2-20:min(a)+abs(min(a)-max(a))//2+20,
                              min(b)+abs(min(b)-max(b))//2-20:min(b)+abs(min(b)-max(b))//2+20,i]))>=30:


                  n_img = normalizeImageIntensityRange(img[min(a)+abs(min(a)-max(a))//2-20:min(a)+abs(min(a)-max(a))//2+20,
                                  min(b)+abs(min(b)-max(b))//2-20:min(b)+abs(min(b)-max(b))//2+20,i])

                  #save the mri slice

                  saveSlice(n_img,str(patients[r])+'_slice_'+str(i),folder+'slices/left_stn/')

                  saveSlice(left_mask[min(a)+abs(min(a)-max(a))//2-20:min(a)+abs(min(a)-max(a))//2+20,
                                  min(b)+abs(min(b)-max(b))//2-20:min(b)+abs(min(b)-max(b))//2+20,i],
                            str(patients[r])+'_slice_'+str(i),folder+'masks/left_stn/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./Training/', help='directory to save Training Slices')
    parser.add_argument('--val_dir', type=str, default='./Validation/', help='directory to save Validation Slices ')
    parser.add_argument('--test_dir', type=str, default='./Testing/', help='directory to save Testing Slices')

    parser.add_argument('--mri_data', type=str, default='./data/mri_crop/', help='directory containing mri data')
    parser.add_argument('--mask_left', type=str, default='./data/mask_left/', help='directory containing left stn mask')
    parser.add_argument('--mask_right', type=str, default='./data/mask_right/', help='directory containing right stn mask')

    parser.add_argument('--n_train', type=int, default=35, help='Number of training images')
    parser.add_argument('--n_val', type=int, default=9, help='Number of validation images')

    opts = parser.parse_args()
    print(opts)

    create_dataset(opts)
