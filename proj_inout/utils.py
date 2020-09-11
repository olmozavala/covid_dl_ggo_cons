import SimpleITK as sitk
import numpy as np

def printImageSummary(image):
    # Classic image functions
    print("---------------------------")
    print("Size: ",image.GetSize())
    print("Origin: ",image.GetOrigin())
    print("Spacing: ", image.GetSpacing())
    print("Direction:", image.GetDirection())
    print("Num Comp:", image.GetNumberOfComponentsPerPixel())
    print("Dimensions: ", image.GetDimension())
    print("IDVal: ", image.GetPixelIDValue())
    print("Type: ",image.GetPixelIDTypeAsString())

def utilsSplitMask(itk_mask):
    np_mask = sitk.GetArrayFromImage(itk_mask)
    np_mask_l1 = np.copy(np_mask)
    np_mask_l2 = np.copy(np_mask)
    np_mask_l3 = np.copy(np_mask)
    np_mask_l1[np_mask != 1] = 0
    np_mask_l2[np_mask != 2] = 0
    np_mask_l3[np_mask != 3] = 0
    itk_mask_1 = sitk.GetImageFromArray(np_mask_l1)
    itk_mask_2 = sitk.GetImageFromArray(np_mask_l2)
    itk_mask_3 = sitk.GetImageFromArray(np_mask_l3)
    return itk_mask_1, itk_mask_2, itk_mask_3