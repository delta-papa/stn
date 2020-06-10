from os.path import isfile, join
import os
import numpy as np
from sampling import reconstruct_volume
from sampling import my_reconstruct_volume
from sampling import load_data_train
#from sampling import load_data_test
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches


import torch
import torch.nn as nn
from LiviaNET import *
from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib

def evaluateSegmentation(gt,pred):
    pred = pred.astype(dtype='int')
    numClasses = np.unique(gt)

    dsc = np.zeros((1,len(numClasses)-1))

    for i_n in range(1,len(numClasses)):
        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==i_n)]=1
        y_c[np.where(pred==i_n)]=1

        dsc[0,i_n-1] = dc(gt_c,y_c)
    return dsc

def numpy_to_var(x):
    torch_tensor = torch.from_numpy(x).type(torch.FloatTensor)

    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    return Variable(torch_tensor)

def crop_volume(mri,mask):

    l_1, l_2, l_3 = np.where(mask==1) #left STN coordinates

    r_1, r_2, r_3 = np.where(mask==2) #right STN coordinates

    left_extent = min(l_1) #location of leftmost voxel of Left STN in x direction
    right_extent = max(r_1) #location of rightmost voxel of the Right STN in x direction

    top_extent = min(l_2) #location of uppermost voxel of Left STN in y direction

    channels = min(l_3)

    mri_roi = mri[left_extent-30:left_extent+90,top_extent-30:top_extent+90,channels-60:channels+60]
    #roi of size 120x120x120

    mask_roi = mask[left_extent-30:left_extent+90,top_extent-30:top_extent+90,channels-60:channels+60]

    return mri_roi,mask_roi

def extract_patches(volume, patch_shape, extraction_step) :

    #ndim = len(volume.shape)
    #npatches = np.prod(patches.shape[:ndim])

    #numPatches = 0
    patchesList = []
    for x_i in range(0,volume.shape[0]-patch_shape[0],extraction_step[0]):
        for y_i in range(0,volume.shape[1]-patch_shape[1],extraction_step[1]):
            for z_i in range(0,volume.shape[2]-patch_shape[2],extraction_step[2]):
                #print('{}:{} to {}:{} to {}:{}'.format(x_i,x_i+patch_shape[0],y_i,y_i+patch_shape[1],z_i,z_i+patch_shape[2]))

                patchesList.append(volume[x_i:x_i + patch_shape[0],
                                          y_i:y_i + patch_shape[1],
                                          z_i:z_i + patch_shape[2]])

    #pdb.set_trace()

    patches = np.concatenate(patchesList, axis=0)
    #return patches.reshape((npatches, ) + patch_shape)
    return patches.reshape((len(patchesList), ) + patch_shape)

def build_set(imageData) :
    patch_shape = (27, 27, 27)
    extraction_step=(5, 5, 5)

    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    imageData_1 = np.squeeze(imageData[0,:,:,:])
    imageData_g = np.squeeze(imageData[1,:,:,:])

    num_classes = len(np.unique(imageData_g))
    x = np.zeros((0, 1, 27, 27, 27))
    y = np.zeros((0, 9, 9, 9))

    #for idx in range(len(imageData)) :
    y_length = len(y)

    label_patches = extract_patches(imageData_g, patch_shape, extraction_step)

    label_patches = label_patches[tuple(label_selector)]

    # Select only those who are important for processing
    valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

    # Filtering extracted patches
    label_patches = label_patches[valid_idxs]

    x = np.vstack((x, np.zeros((len(label_patches), 1, 27, 27, 27))))

    #y = np.vstack((y, np.zeros((len(label_patches), 9, 9, 9))))  # Jose
    y = label_patches
    del label_patches

    # Sampling strategy: reject samples which labels are only zeros
    T1_train = extract_patches(imageData_1, patch_shape, extraction_step)
    x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
    del T1_train

    return x, y

def train_loader(path_mri,path_left,path_right,images,numSamples):

    samplesPerImage = int(numSamples/len(images))

    print(' - Extracting {} samples per image'.format(samplesPerImage))

    X_train = []
    Y_train = []

    for i in range(len(images)):

        imageData_1 = nib.load(path_mri + '/' + images[i]).get_data()
        imageData_left = nib.load(path_left + '/' + images[i]).get_data()

        imageData_right = nib.load(path_right + '/' + images[i]).get_data()

        imageData_g = imageData_left + imageData_right

        num_classes = len(np.unique(imageData_g))

        imageData = np.stack((imageData_1, imageData_g))
        img_shape = imageData.shape

        x_train, y_train = build_set(imageData)

        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)

        x_train = x_train[idx[:samplesPerImage],]
        y_train = y_train[idx[:samplesPerImage],]

        X_train.append(x_train)
        Y_train.append(y_train)

        del x_train
        del y_train

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X = np.concatenate(X_train, axis=0)
    del X_train

    Y = np.concatenate(Y_train, axis=0)
    del Y_train

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], Y[idx], img_shape

def load_data_test(path1, path_left,path_right, imgName):

    extraction_step_value = 9
    imageData_1 = nib.load(path1 + '/' + imgName).get_data()
    imageData_left = nib.load(path_left + '/' + imgName).get_data()
    imageData_right = nib.load(path_right + '/' + imgName).get_data()

    imageData_g = imageData_left + imageData_right

    imageData_1_new = np.zeros((imageData_1.shape[0],imageData_1.shape[1], imageData_1.shape[2] + 2*extraction_step_value))
    imageData_g_new = np.zeros((imageData_1.shape[0],imageData_1.shape[1], imageData_1.shape[2] + 2*extraction_step_value))

    imageData_1_new[:,:,extraction_step_value:extraction_step_value+imageData_1.shape[2]] = imageData_1
    imageData_g_new[:,:,extraction_step_value:extraction_step_value+imageData_g.shape[2]] = imageData_g

    #num_classes = len(np.unique(imageData_g))

    num_classes = 3

    imageData = np.stack((imageData_1_new, imageData_g_new))
    img_shape = imageData.shape

    patch_1 = extract_patches(imageData_1_new, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
    patch_g = extract_patches(imageData_g_new, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

    return patch_1, patch_g, img_shape


def inference(network, moda_1, moda_left,moda_right, imageNames, epoch, folder_save):
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Softmax()
    numClasses = 3  # Move this out
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()

    dscAll = np.zeros((len(imageNames),numClasses-1))  # 1 class is the background!!
    for i_s in range(len(imageNames)):
        patch_1, patch_g, img_shape = load_data_test(moda_1, moda_left,moda_right,imageNames[i_s]) # hardcoded to read the first file. Loop this to get all files
        patchSize = 27
        patchSize_gt = 9
        x = np.zeros((0, 1, patchSize, patchSize, patchSize))
        x = np.vstack((x, np.zeros((patch_1.shape[0], 1, patchSize, patchSize, patchSize))))
        x[:, 0, :, :, :] = patch_1

        pred_numpy = np.zeros((0,numClasses,patchSize_gt,patchSize_gt,patchSize_gt))
        pred_numpy = np.vstack((pred_numpy, np.zeros((patch_1.shape[0], numClasses, patchSize_gt, patchSize_gt, patchSize_gt))))
        totalOp = len(imageNames)*patch_1.shape[0]

        print(totalOp)
        #pred = network(numpy_to_var(x[0,:,:,:,:]).view(1,3,patchSize,patchSize,patchSize))
        for i_p in range(patch_1.shape[0]):
            pred = network(numpy_to_var(x[i_p,:,:,:,:].reshape(1,1,patchSize,patchSize,patchSize)))
            pred_y = softMax(pred)
            pred_numpy[i_p,:,:,:,:] = pred_y.cpu().data.numpy()

            printProgressBar(i_s*((totalOp+0.0)/len(imageNames)) + i_p + 1, totalOp,
                             prefix="[Validation] ",
                             length=15)


        # To reconstruct the predicted volume
        extraction_step_value = 9
        pred_classes = np.argmax(pred_numpy, axis=1)

        pred_classes = pred_classes.reshape((len(pred_classes), patchSize_gt, patchSize_gt, patchSize_gt))

        bin_seg = my_reconstruct_volume(pred_classes,
                                        (img_shape[1], img_shape[2], img_shape[3]),
                                        patch_shape=(27, 27, 27),
                                        extraction_step=(extraction_step_value, extraction_step_value, extraction_step_value))

        bin_seg = bin_seg[:,:,extraction_step_value:img_shape[3]-extraction_step_value]
        gt_left = nib.load(moda_left + '/' + imageNames[i_s]).get_data()
        gt_right = nib.load(moda_right + '/' + imageNames[i_s]).get_data()

        gt = gt_left+gt_right

        img_pred = nib.Nifti1Image(bin_seg, np.eye(4))
        img_gt = nib.Nifti1Image(gt, np.eye(4))

        img_name = imageNames[i_s].split('.nii')
        name = 'Pred_' +img_name[0]+'_Epoch_' + str(epoch)+'.nii.gz'

        namegt = 'GT_' +img_name[0]+'_Epoch_' + str(epoch)+'.nii.gz'

        if not os.path.exists(folder_save + 'Segmentations/'):
            os.makedirs(folder_save + 'Segmentations/')

        if not os.path.exists(folder_save + 'GT/'):
            os.makedirs(folder_save + 'GT/')

        nib.save(img_pred, folder_save + 'Segmentations/'+name)
        nib.save(img_gt, folder_save + 'GT/'+namegt)

        dsc = evaluateSegmentation(gt,bin_seg)

        dscAll[i_s, :] = dsc

    return dscAll

def runTraining(opts):

    samplesPerEpoch = opts.numSamplesEpoch
    batch_size = opts.batchSize

    lr = 0.003
    epoch = opts.numEpochs

    root_dir = opts.root_dir
    model_name = opts.modelName

    ########## Define training data and validation data folders ##########
    moda_1 = root_dir + 'data/mri_crop'
    moda_left = root_dir + 'data/mask_left'
    moda_right = root_dir + 'data/mask_right'



    print(' --- Getting image names.....')
    print(' - Training Set: -')

    """
    If training images and their folder is found then store them in a list else
    raise an error that the folder doesn't exist.

    """
    if os.path.exists(moda_1):
        imageNames_train = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f)) and f.endswith('.img')]
        imageNames_train.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_train)):
            print(' - {}'.format(imageNames_train[i]))
    else:
        raise Exception(' - {} does not exist'.format(moda_1))

    #moda_1_val = root_dir + 'Training/'
    #moda_g_val = root_dir + 'Validation/GT'

    print(' --------------------')
    print(' - Validation Set: -')
    """
    If validation images and their folder is found then store them in a list else
    raise an error that the folder doesn't exist.


    if os.path.exists(moda_1):
        imageNames_val = [f for f in os.listdir(moda_1_val) if isfile(join(moda_1_val, f))]
        imageNames_val.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_val)):
            print(' - {}'.format(imageNames_val[i]))
    else:
        raise Exception(' - {} does not exist'.format(moda_1_val))
    """

    imageNames_val = imageNames_train[:5]

    imageNames_train = imageNames_train[5:]

    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = opts.numClasses
    # Define HyperDenseNet
    # To-Do. Get as input the config settings to create different networks
    if (opts.network == 'liviaNet'):
        print('.... Building LiviaNET architecture....')
        liviaNet = LiviaNet(num_classes)
    else:
        print('.... Building SemiDenseNet architecture....')
        liviaNet = LiviaSemiDenseNet(num_classes)

    '''try:
        hdNet = torch.load(os.path.join(model_name, "Best_" + model_name + ".pkl"))
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        liviaNet.cuda()
        softMax.cuda()
        CE_loss.cuda()

    # To-DO: Check that optimizer is the same (and same values) as the Theano implementation
    optimizer = torch.optim.Adam(liviaNet.parameters(), lr=lr, betas=(0.9, 0.999))

    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    print(' --------- Params: ---------')

    numBatches = int(samplesPerEpoch/batch_size)

    print(' - Number of batches: {} ----'.format(numBatches) )

    dscAll = []
    for e_i in range(epoch):
        liviaNet.train()

        lossEpoch = []

        x_train, y_train, img_shape = train_loader(moda_1, moda_left,moda_right, imageNames_train, samplesPerEpoch) # hardcoded to read the first file. Loop this to get all files. Karthik


        for b_i in range(int(x_train.shape[0]/batch_size)):

            optimizer.zero_grad()
            liviaNet.zero_grad()

            print("Batch number is ",b_i)


            MRIs         = numpy_to_var(x_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:,:])
            Segmentation = numpy_to_var(y_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:])



            segmentation_prediction = liviaNet(MRIs)


            predClass_y = softMax(segmentation_prediction)

            # To adapt CE to 3D
            # LOGITS:
            segmentation_prediction = segmentation_prediction.permute(0,2,3,4,1).contiguous()
            segmentation_prediction = segmentation_prediction.view(segmentation_prediction.numel() // num_classes, num_classes)

            CE_loss_batch = CE_loss(segmentation_prediction, Segmentation.view(-1).type(torch.LongTensor))

            loss = CE_loss_batch

            loss.backward()

            optimizer.step()
            lossEpoch.append(CE_loss_batch.cpu().data.numpy())

            printProgressBar(b_i + 1, numBatches,
                             prefix="[Training] Epoch: {} ".format(e_i),
                             length=15)

            del MRIs
            del Segmentation
            del segmentation_prediction
            del predClass_y

            if not os.path.exists(model_name):
                os.makedirs(model_name)

            np.save(os.path.join(model_name, model_name + '_loss.npy'), dscAll)

            print(' Epoch: {}, loss: {}'.format(e_i,np.mean(lossEpoch)))
        if (e_1%5)==0:
            if not os.path.exists(model_name):
                os.makedirs(model_name)

            torch.save(liviaNet, os.path.join(model_name, "epoch_" + str(e_i+1) + ".pkl"))

        if ((e_i+1)%opts.freq_inference)==0:
            dsc = inference(liviaNet,moda_1, moda_left,moda_right, imageNames_val,e_i, opts.save_dir)
            dscAll.append(dsc)
            print(' Metrics: DSC(mean): {} per class: 1({}) 2({})'.format(np.mean(dsc),np.mean(dsc[:,0]),np.mean(dsc[:,1])))
            if not os.path.exists(model_name):
                os.makedirs(model_name)

            np.save(os.path.join(model_name, model_name + '_DSCs.npy'), dscAll)

            d1 = np.mean(dsc)
            if (d1>0.60):
                if not os.path.exists(model_name):
                    os.makedirs(model_name)

                torch.save(liviaNet, os.path.join(model_name, "Best_" + model_name + ".pkl"))

            if (100+e_i%20)==0:
                 lr = lr/2
                 print(' Learning rate decreased to : {}'.format(lr))
                 for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./Data/MRBrainS/DataNii/', help='directory containing the train and val folders')
    parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
    parser.add_argument('--modelName', type=str, default='liviaNet', help='name of the model')
    parser.add_argument('--network', type=str, default='liviaNet', choices=['liviaNet','SemiDenseNet'],help='network to employ')
    parser.add_argument('--numClasses', type=int, default=3, help='Number of classes (Including background)')
    parser.add_argument('--numSamplesEpoch', type=int, default=1000, help='Number of samples per epoch')
    parser.add_argument('--numEpochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=4, help='Batch size')
    parser.add_argument('--l_rate', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--freq_inference', type=int, default=20, help='Frequency to do the inference on the validation set (i.e., number of epochs between validations)')

    opts = parser.parse_args()
    print(opts)

    runTraining(opts)
