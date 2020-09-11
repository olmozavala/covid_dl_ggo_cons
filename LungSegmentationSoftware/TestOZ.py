from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
from img_viz.medical import MedicalImageVisualizer

input_image = sitk.ReadImage("/data/UM/COVID/Kaggle_Mosmed/studies/study_0001.nii")
segmentation = mask.apply(input_image)  # default model is U-net(R231)

# viz_obj = MedicalImageVisualizer(disp_images=False, output_folder="/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/LungSegmentationSoftware/OUTPUT")
# viz_obj.plot_img_and_ctrs_np(sitk.GetArrayFromImage(input_image), [segmentation],
#                               title="Test Lung seg")