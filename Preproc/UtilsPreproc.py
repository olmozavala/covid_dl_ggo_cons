from scipy.ndimage import binary_fill_holes

from img_viz.medical import MedicalImageVisualizer
from img_viz.constants import SliceMode

import SimpleITK as sitk
import math
import numpy as np
import cv2
from skimage import measure
from scipy.ndimage.measurements import center_of_mass

def compute_center_of_mass(img_np):
    x, y, z = center_of_mass(img_np)
    return int(x), int(y), int(z)

def crop_to_specific_dimensions_from_center(img, ctrs, roi_dims_itk, img_center_of_mass=True):
    """
    Crops a set of images into a ROI. The ROI is defined as a size in cm. The resulting
    image is cropped from the 'center' of the  image
    :param img: itk array of images
    :param ctr
    :param roi_dims_itk
    :param img_center_of_mass
    :return:
    """
    # Get numpy arrays
    img_np = sitk.GetArrayFromImage(img)

    # Get original and desired dimensions
    orig_dims = img_np.shape
    roi_dims = [roi_dims_itk[2], roi_dims_itk[1], roi_dims_itk[0]]

    # Initialize final image and contours containers
    img_crop = np.zeros(roi_dims)

    # These arrays will contain the START and END index for the cropped images
    img_idx = []
    crop_idx = []

    old_origin = img.GetOrigin()
    spacing = img.GetSpacing()
    new_origin = np.zeros(3)

    if img_center_of_mass:
        img_crop_center = compute_center_of_mass(img_np)
        # Not sure why these should be fixed
        img_crop_center = [orig_dims[i] - img_crop_center[i] for i in range(3)]
    else:
        img_crop_center = [int(x / 2) for x in orig_dims]

    # Iterate over the 3 axis
    for axis in range(3):
        # Original and final size
        img_s_orig = orig_dims[axis]
        roi_size_axis = roi_dims[axis]  # Crop size

        # Start at the middle of the image (always)
        img_crop_center_axis = img_crop_center[axis]  # This is to crop from center of mass (TODO It could be improved)
        roi_half_size_axis = roi_size_axis / 2

        # Fixing the image size depending on the center of mass
        img_s = int(min(img_s_orig, min(img_crop_center_axis + roi_half_size_axis, img_s_orig) - max(
            img_crop_center_axis - roi_half_size_axis, 0)))

        # If the size of the image is larger, then we use all the 'cropped' area and reduce the image area
        if img_s >= roi_size_axis:
            crop_idx.append([0, roi_dims[axis]])
            start_img_idx = int(img_crop_center_axis - roi_half_size_axis)
            end_img_idx = int(start_img_idx + roi_size_axis)
            img_idx.append([start_img_idx, end_img_idx])
            new_origin[axis] = spacing[axis] * start_img_idx

        # If the size of the image is smaller, then we use all the 'image' area and reduce the cropped area
        if img_s < roi_size_axis:
            start_img_idx = max(int(img_crop_center_axis - roi_half_size_axis), 0)
            end_img_idx = min(int(img_crop_center_axis + roi_half_size_axis), orig_dims[axis])
            img_idx.append([start_img_idx, end_img_idx])
            start_crop_idx = int(roi_half_size_axis - roi_half_size_axis)
            end_crop_idx = int(start_crop_idx + img_s)
            crop_idx.append([start_crop_idx, end_crop_idx])
            new_origin[axis] = -1 * spacing[axis] * start_crop_idx

    # Setting all the positions as bottom:top, bottom:top, ...
    pos_res = [img_idx[0][0], img_idx[0][1], img_idx[1][0], img_idx[1][1], img_idx[2][0], img_idx[2][1]]
    pos_crop = [crop_idx[0][0], crop_idx[0][1], crop_idx[1][0], crop_idx[1][1], crop_idx[2][0], crop_idx[2][1]]

    print(F'\tCropping this case with the following sizes: (original: {orig_dims})')
    print(F'\tFrom:  {pos_res[0]}:{pos_res[1]},{pos_res[2]}:{pos_res[3]},{pos_res[4]}:{pos_res[5]}')
    print(F'\tTo:    {pos_crop[0]}:{pos_crop[1]},{pos_crop[2]}:{pos_crop[3]},{pos_crop[4]}:{pos_crop[5]}')
    print(F'\tPrev Origin: {old_origin}')
    print(F'\tNew Origin: {new_origin}')

    img_crop[pos_crop[0]:pos_crop[1], pos_crop[2]:pos_crop[3], pos_crop[4]:pos_crop[5]] = \
        img_np[pos_res[0]:pos_res[1], pos_res[2]:pos_res[3], pos_res[4]:pos_res[5]]

    cropped_ctrs = []
    for cur_ctr in ctrs:
        ctr_np = sitk.GetArrayFromImage(cur_ctr)
        ctr_crop = np.zeros(roi_dims)
        ctr_crop[pos_crop[0]:pos_crop[1], pos_crop[2]:pos_crop[3], pos_crop[4]:pos_crop[5]] = \
            ctr_np[pos_res[0]:pos_res[1], pos_res[2]:pos_res[3], pos_res[4]:pos_res[5]]

        cropped_ctrs.append(ctr_crop)

    return img_crop, cropped_ctrs, pos_res, pos_crop, new_origin


def resample_img_itk(in_img, new_spacing: list, interpolator=sitk.sitkLinear, def_value=0) -> sitk.Image:
    ''' Resamples the image to new spacing
    :param img_itk: Input itk image
    :param new_spacing: A list of numbers with the pixel spacing of the image
    :param interpolator: SimpleItk interpolator
    :param def_value:  Default value to use if is required to pad the image
    :return:
    '''

    # Cast image to float
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    in_img = castImageFilter.Execute(in_img)

    original_spacing = in_img.GetSpacing()
    original_size = in_img.GetSize()
    # Computes the new size of the image (required by the itk_filter)
    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]

    # Compute the resampling
    itk_filter = sitk.ResampleImageFilter()

    # "in_img, out size, tranform?, interpolation, out origin, out space, out direction, pixel type id, "
    out_img = itk_filter.Execute(in_img, new_size, sitk.Transform(), interpolator, in_img.GetOrigin(),
                             new_spacing, in_img.GetDirection(), def_value, in_img.GetPixelIDValue())

    return out_img


def resample_to_reference_itk(img_itk, ref_img_itk, interpolator=sitk.sitkLinear, def_value=0):
    '''
    Resamples an itk image to a reference image
    :param img_itk: Input itk image
    :param ref_img_itk: Itk image to use as a reference
    :param interpolator: SimpleItk interpolator
    :param def_value:  Default value to use if is required to pad the image
    :return:
    '''

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img_itk = castImageFilter.Execute(img_itk)

    itk_filter = sitk.ResampleImageFilter()
    itk_filter.SetReferenceImage(ref_img_itk)
    itk_filter.SetDefaultPixelValue(float(def_value))

    itk_filter.SetInterpolator(interpolator)
    out_img_itk = itk_filter.Execute(img_itk)

    return out_img_itk


def optical_flow_interpolation(resampled_ctrs):
    """
    Interpolates resampled contours (by NearestNeighborgs) with Optical flow
    :param resampled_ctrs: Contours that have been resampled previously and we want to 'improve'
    :return:
    """
    # *************** Interpolation between images using OptialFlow (Nearest Neighbor if it fails)
    tot_ctrs = len(resampled_ctrs)
    resampled_ctrs_optical_flow = []
    # Iterate over all the contours
    for ctr_idx in range(tot_ctrs):
        # Resample to reference using nearest neighbor
        tempbk = sitk.GetArrayFromImage(resampled_ctrs[ctr_idx])
        temp = tempbk.copy()
        totslices = temp.shape[0]
        try:
            cur_slice = 0
            # Find the first slice that contains something (skip empty slices)
            while np.sum(temp[cur_slice,:,:], axis=(0,1)) == 0:
                cur_slice += 1
            prev_slice = cur_slice
            next_slice = cur_slice+1

            # Iterate until we are out of slices
            while next_slice < totslices:
                # Verify the next slice is not empty
                if np.sum(temp[next_slice,:,:], axis=(0,1)) == 0:
                    next_slice += 1
                    continue

                # Verify there is difference between layers. Because the interpolation assumes we used
                # Nearest neighbor then it searches for the changes between layers
                if np.sum(temp[prev_slice,:,:] - temp[next_slice,:,:],axis=(0,1)) != 0:
                    dist_intra_slice = next_slice-prev_slice # How many slices do we have in between
                    startLay = prev_slice+int(np.floor(dist_intra_slice/2))-1
                    endLay = next_slice+int(np.floor(dist_intra_slice/2))-1
                    # print('P: {}, N:{}'.format(prev_slice, next_slice))
                    # print('S: {}, E:{}'.format(startLay, endLay))
                    # Do our stuff between prev and next
                    prev = temp[startLay,:,:].astype(np.uint8)*255
                    next = temp[endLay,:,:].astype(np.uint8)*255
                    # Find a contour from the mask
                    ctrs_temp, _= cv2.findContours(prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # Obtain optical flow
                    # print(F' Number of dimensions: {np.ndim(ctrs_temp)} shape: {np.array(ctrs_temp).shape}')
                    ctrs_final = np.array(ctrs_temp[0])[:, 0, :]
                    for ii in range(1,len(ctrs_temp)):
                        ctrs_final = np.concatenate((ctrs_final, np.array(ctrs_temp[ii])[:, 0, :]) ) # Get the proper shape N,2
                    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    orig = ctrs_final.copy() # Copy original position of contours
                    for idSubSlice in np.arange(1,dist_intra_slice): # Move with the flow
                        for idCtr in range(len(ctrs_final)):
                            ctrs_final[idCtr:] = orig[idCtr,:] + (idSubSlice/dist_intra_slice)*flow[orig[idCtr,1], orig[idCtr,0]].T
                        newImg = np.zeros(next.shape)
                        cv2.fillPoly(newImg, pts = [ctrs_final], color=(255,255,255))
                        temp[startLay+idSubSlice,:,:] = newImg

                    prev_slice = endLay
                    next_slice = endLay+1

                next_slice +=1
        except Exception as e:
            print(F"\t\t Warning: Optical Flow Failed. Keeping previous resampled ctr. Error {e}!!!!")
            temp = tempbk # Restore original resampled version with nearest neighbor

        resampled_ctrs_optical_flow.append(sitk.GetImageFromArray(temp))
        resampled_ctrs_optical_flow[ctr_idx] = binaryThresholdImage(resampled_ctrs_optical_flow[ctr_idx], 0.00001)

    return resampled_ctrs_optical_flow


def resample_imgs_and_ctrs(orig_imgs, orig_ctrs, resampling):
    """
    Resamples images and contours to specific resolution.
    :param orig_imgs: itk images to resample
    :param orig_ctrs: itk contours to resample
    :param resampling: array of floats indicating the resampling resolution in mm
    :param ctr_names_tmp: array of names of the ctrs, used only for printing errors
    :param optical_flow_interpolation: bool indicating if we want to do optical flow on the interpolation
    :return:
    """
    tot_imgs = len(orig_imgs)
    tot_ctrs = len(orig_ctrs)
    interpolator = sitk.sitkLinear

    # These 2 vars hold the resampled images and contours
    resampled_imgs = []
    resampled_ctrs = []

    # This is the resampling of the FIRST Image
    resampled_imgs.append(resample_img_itk(orig_imgs[0], resampling, interpolator, 0))
    # resampled_imgs[0] = sizeCorrectionImage(resampled_imgs[0], 6, 168) # Makes the image to size 168 and multiple of 6

    # Resampling all other images with respect to the first image
    valid_idxs = []
    for img_idx in range(1,tot_imgs):
        print(img_idx)
        resampled_imgs.append(resample_to_reference_itk(orig_imgs[img_idx], resampled_imgs[0], interpolator, 0))
        valid_idxs.append(img_idx)

    print("Done!")
    for ctr_idx in range(tot_ctrs):
        # Contours are always resampled using Nearest Neighbor
        resampled_ctrs.append(resample_to_reference_itk(orig_ctrs[ctr_idx], resampled_imgs[0],
                                                       interpolator=sitk.sitkNearestNeighbor, def_value=0))

    return resampled_imgs, resampled_ctrs


def normalize_to_percentiles(imgs, minperc=1, maxperc=99):
    """
    Normalizes an array of images to 0 to 1 using the specified percentiles
    :param imgs:
    :param minperc:
    :param maxperc:
    :return:
    """
    out = []

    normalizationFilter = sitk.IntensityWindowingImageFilter()

    # This part normalizes the images
    for idx in range(len(imgs)):
        img = imgs[idx]
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        # Gets the value of the specified percentiles
        upperPerc = np.percentile(array, maxperc)
        lowerPerc = np.percentile(array, minperc)

        normalizationFilter.SetOutputMaximum(1.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        floatImg= sitk.Cast(img, sitk.sitkFloat32) # Cast to float

        # ALL images get normalized between 0 and 1
        outNormalization = normalizationFilter.Execute(floatImg) #Normalize to 0-1
        out.append(outNormalization)

        # If you want to see the differences before and after normalization
        # utilsviz.drawSeriesItk(floatImg, slices=[90], title='', contours=[], savefig='', labels=[])
        # utilsviz.drawSeriesItk(out[-1], slices=[90], title='', contours=[], savefig='', labels=[])
    return out

def binaryThresholdImage(img, lowerThreshold):
    '''
    Obtains a binary image.
    :param img: Itk image
    :param lowerThreshold: Lower threshold to use as valid value.
    :return:
    '''

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded

def sizeCorrectionBoundingBox(bbox, newSize, factor):
    # adapt the start index of the ROI to the manual bounding box size
    # (assumes that all ROIs are smaller than newSize pixels in length and width)
    # correct the start index according to the new size of the bounding box
    start = bbox[0:3]
    start = list(start)
    size = bbox[3:6]
    size = list(size)
    start[0] = start[0] - math.floor((newSize - size[0]) / 2)
    start[1] = start[1] - math.floor((newSize - size[1]) / 2)

    # check if BB start can be divided by the factor (essential if ROI needs to be extracted from non-isotropic image)
    if (start[0]) % factor != 0:
        cX = (start[0] % factor)
        newStart = start[0] - cX
        start[0] = int(newStart)

    # y-direction
    if (start[1]) % factor != 0:
        cY = (start[1] % factor)
        start[1] = int(start[1] - cY)

    size[0] = newSize
    size[1] = newSize

    return start, size

def thresholdImage(img, lowerValue, upperValue, outsideValue):

    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetUpper(upperValue)
    thresholdFilter.SetLower(lowerValue)
    thresholdFilter.SetOutsideValue(outsideValue)

    out = thresholdFilter.Execute(img)
    return out

def getBoundingBox(img):
    '''
    Gets the bbox with values from an image
    :param img:
    :return:
    '''

    masked = binaryThresholdImage(img, 0.1)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(masked)

    bb = statistics.GetBoundingBox(1)

    return bb

def getCroppedIsotropicImgsOZ(imgs, ctrs, img_size=168):
    '''
    Crops the images with the intersection of Sag, Transversal and Coronal
    :param output_folder:
    :param imgs: VERY important it is assumed that the first 3 images are tra, cor, sag
    :param ctrs:
    :return:
    '''
    tot_imgs = len(imgs)
    tot_ctrs = len(ctrs)

    assert tot_imgs >= 3, 'Less than 3 images to compute the isotropic croping'
    masks = []
    # Obtain a mask for each image (ones everything above 0)
    for img_idx in range(3): # We NEED the 3 first images (AX,SAG,COR) or it will not work
        masks.append(binaryThresholdImage(imgs[img_idx], 0.0001))

    # WARNING this part assumes the order tra, cor, sag
    mask_cor_tra = sitk.Multiply(masks[0], masks[1]) # Intersection tra and cor
    mask_all = sitk.Multiply(masks[2], mask_cor_tra) # Intersection tra, cor and sag
    bbox = getBoundingBox(mask_all)

    # Obtains the start positions and size of the image to cut (size should always be 168^3
    start, size = sizeCorrectionBoundingBox(bbox, newSize=img_size , factor=6)

    roi_imgs = []
    roi_ctrs = []

    # Cuts all the images to the ROI
    # print([x.GetSize() for x in imgs])
    # utilsviz.drawMultipleSeriesItk(imgs, slices=SliceMode.MIDDLE, contours=mask_all)
    for img_idx in range(tot_imgs):
        roi_imgs.append( sitk.RegionOfInterest(imgs[img_idx], [size[0], size[1], size[2]], [start[0], start[1], start[2]]))
    # utilsviz.drawMultipleSeriesItk(roi_imgs, slices=SliceMode.MIDDLE)

    for ctr_idx in range(tot_ctrs):
        roi_ctrs.append( sitk.RegionOfInterest(ctrs[ctr_idx], [size[0], size[1], size[2]], [start[0], start[1], start[2]]))


    return roi_imgs, roi_ctrs, start, size

def getLargestConnectedComponents(img):

    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels+1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent

def getLargestConnectedComponentsBySliceAndFillHoles(img):
    '''
    Obtains the largest connected component, but in a 2D fassion slice by slice
    :param img:
    :return:
    '''
    for ii in range(img.shape[0]):# Iterate each slice
        if np.sum(img[ii,:,:]) > 0:
            # import matplotlib.pyplot as plt
            # plt.imshow(img[ii,:,:])
            # plt.show()
            # Obtain connected component
            all_labels = measure.label(img[ii,:,:], background=0)
            if all_labels.max() > 1: # Check if there is more than one connected component
                larg_label = -1
                larg_val = -1
                for jj in range(1,all_labels.max()+1): # Select the largest component
                    cur_sum = np.sum(all_labels == jj)
                    if  cur_sum > larg_val:
                        larg_label = jj
                        larg_val = cur_sum
                # Remove everything except the largest component
                all_labels[all_labels != larg_label] = 0
                all_labels[all_labels == larg_label] = 1
                img[ii,:,:] = all_labels
            # Fill the holes
            img[ii,:,:] = binary_fill_holes(img[ii,:,:])
            # plt.imshow(img[ii,:,:])
            # plt.show()

    return img

def shift3D(A, size, axis):
    dims = A.shape
    if axis==0:
        if size > 0:
            B = np.lib.pad(A[:dims[1]-size,:,:], ((size, 0), (0, 0), (0,0)), 'edge')
        else:
            B = np.lib.pad(A[-size:, :,:], ((0, -size), (0, 0), (0,0)), 'edge')
    if axis==1:
        if size > 0:
            B = np.lib.pad(A[:,:dims[1]-size, :], ((0, 0), (size, 0), (0,0)), 'edge')
        else:
            B = np.lib.pad(A[:,-size:, :], ((0, 0), (0, -size), (0,0)), 'edge')
    if axis==2:
        if size > 0:
            B = np.lib.pad(A[:,:,:dims[1]-size], ((0, 0), (0,0), (size, 0)), 'edge')
        else:
            B = np.lib.pad(A[:,:,-size: ], ((0, 0), (0,0), (0, -size)), 'edge')
    return B


