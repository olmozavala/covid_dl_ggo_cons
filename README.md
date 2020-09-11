# Software

## Lung segmentation
[https://github.com/JoHof/lungmask] (https://github.com/JoHof/lungmask)

# Data

## Summary

### Kaggle Mosmed [kaggle link](https://www.kaggle.com/xuehaihe/covidct)

50 annotated 3D CT scans with ROI. Each ROI indicates
(ground-glass opacifications and consolidations).
Image sizes are fixed horizontally at **512x512xZ**. 

### Medical Segmentation 1 [link] (http://medicalsegmentation.com/covid19/)

100 annotated 2D CT slices. The segmented masks indicate areas of 
ground-glass (value = 1), consolidation (value = 2), 
pleural effusion (value = 3).
Lung masks available but not being used (check the website)

### Medical Segmentation 2 [link] (http://medicalsegmentation.com/covid19/)

9 annotated 3D CT scans. The segmented masks indicate areas of 
ground-glass (value = 1), consolidation (value = 2), 
pleural effusion (value = 3). It also includes lung masks
Image sizes are fixed horizontally at **630x630xZ**. 


### Radiopedia (Zenodo) 2 [link] (https://zenodo.org/record/3757476#.X06lFXZKhwF)

20 annotated 3D CT scans with ROI. Each ROI indicates
 (left and right lung and infection).
Image sizes are not fixed in any direction. 

# Report
This research's original objective was to develop a Deep Learning (DL)
approach for risk stratification of patients with COVID-19. That system
could help identify patients that will need intensive care unit (ICU)
care in advance.

The development of such a DL system requires a dataset that includes
outcome information (recovered, deceased, or required ICU) about the
patient. So far, we have not found any public dataset with such
information. The IRB approval of the retrospective review of
radiographic images acquired from Covid-19 patients at the University of
Miami is still pending.

Without patient risk information, we decided to modify the grant's
short-term scope and pursue research on DL systems capable of
automatically segmenting ground-glass opacifications/opacities (GGO) or
areas of consolidation from CT images. Besides the research on DL
systems, my main contribution was collecting, analyzing, and
preprocessing public databases containing CT and Chest X-Ray images.

Table 1 contains a summary of the databases that were collected and
analyzed by the group. My research focused on the five databases with
ids 3, 4, 5, and 18; because they contain segmentations of interest (GGO
or consolidation). These databases were preprocessed in the following
way:

1.  CT images were resampled to a homogenized pixel resolution of .9 x
    .9 x 4.5 mm.
2.  Images were cropped from the center to obtain a grid of size 320 x
    320 x 80. This step is essential because many DL algorithms still
    require fixed-size inputs.
3.  Image intensities were normalized to a range of 0 to 1 from their
    1st and 99th percentiles.
4.  Automatically segmented the lungs for the fifty patients acquired
    from database No. 18 (Kaggle Mosmed). The segmentation is computed
    using the publicly available DL system from Hofmanninger et
    al.^[^1]^ This system segments the right and left lungs from CT
    images robustly; the details of this algorithm are described in
    their paper \"*Automatic lung segmentation in routine imaging is
    primarily a data diversity problem, not a methodology problem."*
5.  The most extensive database that we found contains 50 CT segmented
    cases. Still, this database only labels the segmented area as "each
    ROI indicates GGO and consolidations," and we do not have any means
    to know which regions of the segmented area correspond to each
    specific label. We modified all the databases' segmented labels to
    only *affected *(GGO or consolidation) and lung.
6.  Each database has a unique naming convention on its images and
    labels. We organized the data into two continuous databases with
    uniform naming conventions and file formats as follows:

    -   The first database contains 78 CT images saved with the *nearly
        > raw raster data *(nrrd) file format. Each CT image is saved on
        > its folder with a sequential numbering like this:

> ![](Pictures/100000000000009100000121C0EE15BFCA526AAE.png){width="0.7681in"
> height="1.5209in"}

1.  -   The second database contains 2429 CT slices. These slices are
        > the ones with some segmented areas of interest. In this case,
        > the images are saved with the png image file format, and the
        > segmented regions of interests, *lesions* and lungs, are saved
        > separated, as follows:

> ![](Pictures/10000000000000D10000015DB19A4A45D58BEB8E.png){width="0.9756in"
> height="1.6283in"}

Figure 1 shows an example of automatic generations of lung masks using
the Hofmanninger algorithm.

![](Pictures/10000000000004C4000004B9AEF35454D488DD18.png){width="3.7362in"
height="3.7063in"}

Fig 1. Results of the automatic segmentation of lung masks.

Red is the right lung and green is the left lung.

Figure 2 shows an example of the normalization, resampling, and cropping
process been applied to CT images.

![](Pictures/10000000000003D5000003CDCCB72B27C58DCF97.png){width="3.0264in"
height="3.0028in"}
![](Pictures/10000000000004E50000052A3FF2897B5C14DC94.png){width="2.8409in"
height="3.0083in"}

Fig 2. Example image showing the results of the proposed preprocessing
algorithm (normalization, resampling, and cropping). On the left, the
original CT image, on the right, the post-processed version.

Regarding the DL system, I tested custom versions of 3D and 2D U-Net
architectures. In both architectures, the dataset was split on 80% for
training, 10% for validation, and 10% for testing. These systems were
developed using the Tensorflow and Keras frameworks in Python. The
networks are trained with Stochastic Gradient Descent as the
optimization method, and the modified negative Sørensen--Dice
coefficient (DSC) is explained below as the loss function. Both systems
were trained up to 2000 epochs. The best validation parameters are saved
on each training session, and the parameters with the minimum validation
loss were used to compute our final results.

To improve the custom U-Net networks' training, we compute the loss
function (negative DSC) only on the lungs' pixels and not on the whole
CT image. The proposed loss function on both networks (3D and 2D) is:

![](Pictures/10000201000000DE0000007720BE7AD4CE1EF209.png){width="1.5098in"
height="0.8098in"}

N is the total number of voxels **inside the lungs**, p~i~ the voxel
values predicted by the network, and t~i~ the actual voxel values of the
ROI (GGO or consolidation).

Figure 3 shows a schema of the proposed modified 3D U-Net.

![](Pictures/10000000000007EE000001BDBC657AABCA86DAE8.png){width="6.5in"
height="1.4307in"}

Fig 3. Proposed 3D U-Net for the automatic segmentation of GGO or
consolidation areas.

The proposed 2D U-Net follows the same 3D architecture. The difference
is that the input and output sizes are 2D with dimensions 320 x 320 and
that the initial number of image filters is set to 16, displayed in
figure 4.

![](Pictures/10000000000007EE000001BDBC657AABCA86DAE8.png){width="6.5in"
height="1.4307in"}

Fig 4. Proposed 2D U-Net for the automatic segmentation of GGO or
consolidation areas.

[]{#anchor}Results

For 3D, the network obtains a DSC of **0.42 ± 0.23 , 0.43 ± 0.25,
**and** 0.64** ** ± 0.11 **for the training, validation, and test sets
respectively. From these results, we can conclude that the proposed
system can broadly identify the GGO and consolidation areas in the lung.
But it requires more training examples to get better statistics; let us
remember that for the 3D network, we only have 79 cases, from which only
63 are used for training. Having a higher DSC for the test dataset does
not mean that the network will achieve higher DSC for unseen cases. It
only means that for the very few test cases, the system performed
satisfactorily. A more in-depth analysis is required to understand what
characteristics do the instances where the network performs poorly have.
Figure 5 shows three examples of the segmented regions of interest, one
case from each of the training, validation, and test sets. Table 2 in
the appendix has all the obtained DSC for all the patients; the data
split is the following:

**Train examples (total:63) :\[\'Case-0000\' \'Case-0001\' \'Case-0002\'
\'Case-0005\' \'Case-0006\' \'Case-0007\' \'Case-0008\' \'Case-0009\'
\'Case-0012\' \'Case-0013\' \'Case-0014\' \'Case-0016\' \'Case-0017\'
\'Case-0018\' \'Case-0019\' \'Case-0020\' \'Case-0021\' \'Case-0022\'
\'Case-0024\' \'Case-0025\' \'Case-0026\' \'Case-0027\' \'Case-0028\'
\'Case-0029\' \'Case-0030\' \'Case-0031\' \'Case-0032\' \'Case-0033\'
\'Case-0034\' \'Case-0036\' \'Case-0037\' \'Case-0038\' \'Case-0039\'
\'Case-0040\' \'Case-0041\' \'Case-0042\' \'Case-0043\' \'Case-0045\'
\'Case-0046\' \'Case-0047\' \'Case-0049\' \'Case-0050\' \'Case-0053\'
\'Case-0054\' \'Case-0055\' \'Case-0056\' \'Case-0057\' \'Case-0058\'
\'Case-0059\' \'Case-0060\' \'Case-0061\' \'Case-0062\' \'Case-0064\'
\'Case-0065\' \'Case-0067\' \'Case-0069\' \'Case-0070\' \'Case-0071\'
\'Case-0072\' \'Case-0074\' \'Case-0075\' \'Case-0077\'
\'Case-0078\'\]**

**Validation examples (total:8) :\[\'Case-0004\' \'Case-0010\'
\'Case-0011\' \'Case-0015\' \'Case-0023\' \'Case-0052\' \'Case-0073\'
\'Case-0076\'\]:**

**Test examples (total:8) :\[\'Case-0003\' \'Case-0035\' \'Case-0044\'
\'Case-0048\' \'Case-0051\' \'Case-0063\' **

![](Pictures/100000000000027C000002998A1E103246FD7490.jpg){width="2.0819in"
height="2.1744in"}![](Pictures/100000000000027C000002996D8B52C99B469AB6.jpg){width="2.1189in"
height="2.2165in"}![](Pictures/100000000000027C00000299F142196B74EEC493.jpg){width="2.0898in"
height="2.1874in"}

Fig. 5 Examples of the segmentation obtained with the proposed network.
From left to right, examples from the training, validation, and test
sets.

For 2D, the network obtains DSC of **0.68 ± 0.24, 0.52 ± 0.29, **and**
0.60 ±** **0.28**for the training, validation, and test sets
respectively. It is important to mention that even when the 2D dataset
is also split into training, validation, and test, some slices in the
test set may come from a patient with some training slices on the
training set. To obtain a fair comparison between the two systems, we
will need to separate complete 3D cases for the test set in 2D. Figure 6
shows four examples, two from the validation set and two from the test
set. These examples have DSC close to the mean DSC obtained in each set

![](Pictures/10000000000002770000029576686FD795EE506C.png){width="2.6154in"
height="2.7319in"}![](Pictures/100000000000027C0000029994C55D16BA892123.jpg){width="2.6138in"
height="2.7339in"}

![](Pictures/100000000000027600000293D099408AEABDBADF.png){width="2.672in"
height="2.7925in"}![](Pictures/100000000000027600000293FB4B225DFD3C3FC0.png){width="2.6575in"
height="2.7811in"}

Fig 6. Four examples of the segmentations obtained from the proposed 2D
U-Net. The two images on the top correspond to examples from the
validation set which are close to the DSC mean of that set (0.52). The
two images on the bottom are examples from the test set with DSC similar
to the mean of that set (0.598).

[]{#anchor-1}Conclusions

We have created an extensive database of 3D and 2D COVID-19 CT examples
that is suitable for machine learning use. Additionally, we have
developed two DL systems capable of identifying GGO or consolidation
areas automatically. These two systems still require to be analyzed in
more detail. The considerations used while making the segmentations of
the regions of interest seem significantly different between the
databases. A consensus is needed to evaluate the systems better; an
intra-expert variability study could show how distant the segmentations
made from two radiologists are.

Table 1. Summary of the analyzed COVID databases. More information about
each of these databases can be obtained in this link:
[*https://docs.google.com/spreadsheets/d/1vbQRYLpxNBpbVXhmdtmxpJf\_hwd2b1TCyYOvhXs7OsA/edit?usp=sharing*](https://docs.google.com/spreadsheets/d/1vbQRYLpxNBpbVXhmdtmxpJf_hwd2b1TCyYOvhXs7OsA/edit?usp=sharing)

+---------+---------+---------+---------+---------+---------+---------+
| ID      | Name    | \#      | Size    | Image   | Labeled | File    |
|         |         | Cases   |         | Type    | type    | Format  |
+---------+---------+---------+---------+---------+---------+---------+
| 1       | UCSD    | 275     |         | CT      | Only    | png     |
|         |         |         |         |         | some    |         |
|         |         |         |         |         | info    |         |
|         |         |         |         |         | about   |         |
|         |         |         |         |         | the     |         |
|         |         |         |         |         | patient |         |
|         |         |         |         |         | .       |         |
+---------+---------+---------+---------+---------+---------+---------+
| 2       | IEEE    | 210     |         | X-ray   | Lungs   | png     |
|         | X-Ray   |         |         |         | mask,   |         |
|         |         |         |         |         | COVID   |         |
|         |         |         |         |         | severit |         |
|         |         |         |         |         | y       |         |
|         |         |         |         |         | score,  |         |
|         |         |         |         |         | BBOX    |         |
+---------+---------+---------+---------+---------+---------+---------+
| 3       | Medical | 40      |         | CT      | Segment | Nifti   |
|         | segment | patient |         |         | ed.     | (2d)    |
|         | ation   | s       |         |         | Ground- |         |
|         |         | 100     |         |         | glass,  |         |
|         |         | axial   |         |         | consoli |         |
|         |         | CT      |         |         | dation, |         |
|         |         |         |         |         | pleural |         |
|         |         |         |         |         | effusio |         |
|         |         |         |         |         | n       |         |
|         |         |         |         |         | (labels |         |
|         |         |         |         |         | 1, 2,   |         |
|         |         |         |         |         | and 3)  |         |
+---------+---------+---------+---------+---------+---------+---------+
| 4       | Medical | 9       |         | CT      | Segment |         |
|         | segment | Patient |         |         | ed.     |         |
|         | ation   | s       |         |         | Ground- |         |
|         | V2      | (3D)    |         |         | glass,  |         |
|         |         |         |         |         | consoli |         |
|         |         |         |         |         | dation, |         |
|         |         |         |         |         | pleural |         |
|         |         |         |         |         | effusio |         |
|         |         |         |         |         | n       |         |
|         |         |         |         |         | (labels |         |
|         |         |         |         |         | 1, 2,   |         |
|         |         |         |         |         | and 3)  |         |
+---------+---------+---------+---------+---------+---------+---------+
| 5       | Kaggle  | 20 (3D  | 7 GB    | CT      | Segment | NifTi   |
|         | 1       | CT)     |         |         | ed.     |         |
|         | Radiope |         |         |         | Lung    |         |
|         | dia     |         |         |         | and     |         |
|         | Zenodo  |         |         |         | infecti |         |
|         |         |         |         |         | on      |         |
|         |         |         |         |         | mask    |         |
+---------+---------+---------+---------+---------+---------+---------+
| 6       | Kaggle  | 219     | 1 GB    | X-Ray   | COVID   | png     |
|         | 2       | Positiv |         |         | vs.     |         |
|         |         | e,      |         |         | Healthy |         |
|         | COVID-1 | 1341    |         |         | vs.     |         |
|         | 9       | healthy |         |         | Pneumon |         |
|         | RADIOGR | ,       |         |         | ia      |         |
|         | APHY    | 1345    |         |         |         |         |
|         |         | viral   |         |         |         |         |
|         |         | pneumon |         |         |         |         |
|         |         | ia      |         |         |         |         |
|         |         | (2D)    |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 7       | Kaggle  | 1576    | 1 GB    | X-Ray   | CSV     | png     |
|         | 3       | Normal, |         |         | file    |         |
|         |         | \~4k    |         |         | indicat |         |
|         |         | Pneumon |         |         | ing:    |         |
|         |         | ia      |         |         | Stress- |         |
|         |         | (2D)    |         |         | smoking |         |
|         |         |         |         |         | ,       |         |
|         |         |         |         |         | Virus   |         |
|         |         |         |         |         | (COVID, |         |
|         |         |         |         |         | SARS),  |         |
|         |         |         |         |         | Bacteri |         |
|         |         |         |         |         | a       |         |
|         |         |         |         |         | (strept |         |
|         |         |         |         |         | ococcus |         |
|         |         |         |         |         | )       |         |
+---------+---------+---------+---------+---------+---------+---------+
| 8       | Kaggle  | 6432    | 2 GB    | X-Ray   | COVID   | jpg     |
|         | 4       |         |         |         | vs.     |         |
|         |         | (2D)    |         |         | Healthy |         |
|         |         |         |         |         | vs.     |         |
|         |         |         |         |         | Pneumon |         |
|         |         |         |         |         | ia      |         |
+---------+---------+---------+---------+---------+---------+---------+
| 9       | Kaggle  | 600     | 2 GB    | X-Ray   | COVID   | jpg     |
|         | 5 GAN   | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
|         |         |         |         |         | vs.     |         |
|         |         |         |         |         | Pneumon |         |
|         |         |         |         |         | ia      |         |
+---------+---------+---------+---------+---------+---------+---------+
| 10      | Kaggle  | 300     | 1.42    | X-Ray   | Labels: | jpg     |
|         | 6       | (2D)    |         |         | atelect |         |
|         |         |         |         |         | asis,   |         |
|         |         |         |         |         | cardiom |         |
|         |         |         |         |         | egaly,  |         |
|         |         |         |         |         | consoli |         |
|         |         |         |         |         | dation, |         |
|         |         |         |         |         | COVID-1 |         |
|         |         |         |         |         | 9,      |         |
|         |         |         |         |         | edema,  |         |
|         |         |         |         |         | effusio |         |
|         |         |         |         |         | n,      |         |
|         |         |         |         |         | emphyse |         |
|         |         |         |         |         | ma,     |         |
|         |         |         |         |         | fibrosi |         |
|         |         |         |         |         | s,      |         |
|         |         |         |         |         | etc.    |         |
+---------+---------+---------+---------+---------+---------+---------+
| 11      | Kaggle  | 3000    | 1.97    | X-Ray   | COVID   | jpg     |
|         | 7       | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
+---------+---------+---------+---------+---------+---------+---------+
| 12      | Kaggle  | 18k     | 4 GB    | CT      | COVID   | jpg     |
|         | Abu     | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
+---------+---------+---------+---------+---------+---------+---------+
| 13      | Kaggle  | 13k     | 4 GB    | X-Ray   | No      | jpg     |
|         | Sethu   | (2D)    |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 14      | Kaggle  | 200     | 1.4 GB  | X-Ray   | COVID   | Tiff    |
|         | Chris   | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
|         |         |         |         |         | vs.     |         |
|         |         |         |         |         | Pneumon |         |
|         |         |         |         |         | ia      |         |
+---------+---------+---------+---------+---------+---------+---------+
| 15      | Kaggle  | 377     | 22 GB   | CT      | COVID   | Tiff    |
|         | Mohamma | patient |         |         | vs.     |         |
|         | d       | s/      |         |         | Healthy |         |
|         |         | 63849   |         |         |         |         |
|         |         | (2D)    |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 16      | Kaggle  | 2500    | 1 GB    | CT      | COVID   | jpg     |
|         | Anas    | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
|         |         |         |         |         | vs.     |         |
|         |         |         |         |         | Pneumon |         |
|         |         |         |         |         | ia      |         |
+---------+---------+---------+---------+---------+---------+---------+
| 17      | BIMCV   | 1400    | 70 GB   | X-Ray   |         |         |
|         |         | X-Ray   |         | and CT  |         |         |
|         |         | and 163 |         |         |         |         |
|         |         | CT (2D) |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 18      | Kaggle  | 50 (3D) | 3 GB    | CT      | Segment | NifTi   |
|         | Mosmed  |         |         |         | ed.     |         |
|         |         |         |         |         | ROI     |         |
|         |         |         |         |         | (Ground |         |
|         |         |         |         |         | -glass  |         |
|         |         |         |         |         | or      |         |
|         |         |         |         |         | consoli |         |
|         |         |         |         |         | dation) |         |
+---------+---------+---------+---------+---------+---------+---------+
| 19      | Kaggle  | 275     | 55 MB   | CT      | No      | png     |
|         | Xuehai  | (2D)    |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 20      | Kaggle  | 210     | 407 MB  | CT      | COVID   | png     |
|         | Plamen  | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
+---------+---------+---------+---------+---------+---------+---------+
| 21      | Kaggle  | 100     | 800 MB  | CT      | Segment | h5      |
|         | Md      | (2D)    |         |         | ed.     |         |
|         | Awsafur |         |         |         | Ground- |         |
|         |         |         |         |         | glass,  |         |
|         | (same   |         |         |         | consoli |         |
|         | as id   |         |         |         | dation, |         |
|         | 3)      |         |         |         | pleural |         |
|         |         |         |         |         | effusio |         |
|         |         |         |         |         | n       |         |
|         |         |         |         |         | (labels |         |
|         |         |         |         |         | 1, 2,   |         |
|         |         |         |         |         | and 3)  |         |
+---------+---------+---------+---------+---------+---------+---------+
| 22      | Figshar | 300     | 98 MB   | CT      | COVID   | png     |
|         | e       | (2D)    |         |         | vs.     |         |
|         |         |         |         |         | Healthy |         |
+---------+---------+---------+---------+---------+---------+---------+

Table 2. 3D DSC obtained for all the cases using the custom 3D U-Net
network.

3D\_DSC

Case-00000.616

Case-00010.672

Case-00020.186

Case-00030.751

Case-00040.255

Case-00050.009

Case-00060.686

Case-00070.390

Case-00080.229

Case-00090.209

Case-00100.766

Case-00110.409

Case-00120.255

Case-00130.049

Case-00140.361

Case-00150.558

Case-00160.451

Case-00170.647

Case-00180.709

Case-00190.406

Case-00200.538

Case-00210.728

Case-00220.320

Case-00230.469

Case-00240.555

Case-00250.537

Case-00260.121

Case-00270.207

Case-00280.016

Case-00290.586

Case-00300.260

Case-00310.047

Case-00320.002

Case-00330.425

Case-00340.340

Case-00350.759

Case-00360.725

Case-00370.570

Case-00380.203

Case-00390.379

Case-00400.275

Case-00410.062

Case-00420.286

Case-00430.177

Case-00440.463

Case-00450.406

Case-00460.084

Case-00470.486

Case-00480.646

Case-00490.163

Case-00500.624

Case-00510.763

Case-00520.727

Case-00530.584

Case-00540.611

Case-00550.000

Case-00560.711

Case-00570.583

Case-00580.295

Case-00590.306

Case-00600.703

Case-00610.573

Case-00620.403

Case-00630.572

Case-00640.686

Case-00650.771

Case-00660.646

Case-00670.657

Case-00680.547

Case-00690.751

Case-00700.695

Case-00710.382

Case-00720.524

Case-00730.000

Case-00740.737

Case-00750.461

Case-00760.286

Case-00770.524

Case-00780.608

**AVG0.445**

[^1]:  (2020, August 20). Automatic lung segmentation in routine imaging
    is primarily a \.... Retrieved September 4, 2020, from
    [*https://eurradiolexp.springeropen.com/articles/10.1186/s41747-020-00173-2*](https://eurradiolexp.springeropen.com/articles/10.1186/s41747-020-00173-2)

