from spottunet import paths

import torch
from scipy import misc
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import logging
from medpy import metric


def _eval_dice(gt_y, pred_y, detail=False):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "bg",
        "1": "CZ",
        "2": "prostate"
    }

    dice = []
    #for j in range(gt_y.shape[2]):

    for cls in range(1,2):

        #gt = np.zeros((gt_y.shape[0],gt_y.shape[1]))
        #pred = np.zeros((pred_y.shape[0],pred_y.shape[1]))#np.zeros(pred_y.shape)

        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)#np.zeros(pred_y.shape)

        #gt[gt_y[:,:,j] == cls] = 1
        #pred[pred_y[:,:,j] == cls] = 1
        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1


        dice_this = (2*np.sum(gt*pred))/(np.sum(gt)+np.sum(pred))
        '''
        if (np.sum(gt)+np.sum(pred)==0):
            dice_this=1.0
        '''
        dice.append(dice_this)


    if detail is True:
        #print ("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
        logging.info("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
    return dice

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def _eval_average_surface_distances(reference, result):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)



def compute_metrics_msm(ids,predict):

    dice = []
    asd = []
    for ind, id1 in enumerate(ids):
        file_image_path = os.path.join(paths.MSM_DATA_PATH,id1[0])

        file_label_path = os.path.join(paths.MSM_DATA_PATH,id1[1])

        itk_image = sitk.ReadImage(
            file_image_path)  # os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
        itk_mask = sitk.ReadImage(file_label_path)
        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)
        binary_mask = np.ones(mask.shape)
        mean = np.sum(image * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
        image = (image - mean) / std
        mask[mask == 2] = 1
        image = image.transpose([1, 2, 0])
        mask = mask.transpose([1, 2, 0])
        end_shape = (384, 384)
        preds = np.zeros(mask.shape)
        frame_list = [kk for kk in range(1, image.shape[2] - 1)]
        for ii in range(image.shape[2]):
            vol = np.zeros([1, 3, end_shape[0], end_shape[1]],dtype=np.float32)

            for idx, jj in enumerate(frame_list[ii: ii + 1]):
                vol[0, :, :, :] = image[..., jj - 1: jj + 2].transpose(
                    [2, 0,
                     1])  # vol = image[0, jj - 1: jj + 2,:,:].copy()vol[idx, ...] = image[..., jj - 1: jj + 2].copy()
            torch.cuda.empty_cache()
            images = torch.from_numpy(vol)
            outputs = predict(images)

            for idx, jj in enumerate(frame_list[ii: (ii + 1)]):
                preds[..., jj] = (outputs > 0.5).squeeze().copy()
        processed_preds = _connectivity_region_analysis(preds)
        dice_subject = _eval_dice(mask, processed_preds)
        asd_subject = _eval_average_surface_distances(mask, processed_preds)

        dice.append(dice_subject)
        asd.append(asd_subject)

    dice_avg = np.mean(dice, axis=0).tolist()[0]
    asd_avg = np.mean(asd)

    print("dice_avg_student %.4f" % (dice_avg))
    print("asd_avg_student %.4f" % (asd_avg))

    return {'dice':float(dice_avg),'asd':float(asd_avg)}


