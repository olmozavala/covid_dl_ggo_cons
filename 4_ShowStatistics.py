import numpy as np
import re
import pylab
from os.path import getmtime
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
import matplotlib.image as mpimg
import imageio

def main():
    ''' What this code does it reads the splits from the models and the DSC from the proper folders, and displays
     the training and validation DSC for each of them'''

    # ======================= FOR KIDNEYS DELETE ==================
    # model = "COVID_SingleStream_2DUNet_2020_09_10_18_53"
    # metrics_csv = "/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_2DUNet/DSCs/Classification_DSC.csv"
    # splits = readingSplitsFromModel("/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_2DUNet/Splits",[model])
    model = "COVID_SingleStream_3DUNet_2020_09_09_19_01"
    metrics_csv = "/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_3DUNet/DSCs/Classification_DSC.csv"
    splits = readingSplitsFromModel("/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_3DUNet/Splits",[model])
    df = pd.read_csv(metrics_csv, index_col=0)
    # for cur_resolution in ['2D_DSC']:
    for cur_resolution in ['3D_DSC']:
        hists = []
        avg = []
        std = []
        for kk, cur_subset in enumerate(['train', 'validation', 'test']):
            indexes = numToCases(splits[F'{model}_{cur_subset}'])
            hists.append(df.filter(items=indexes, axis=0))
            avg.append(F'{hists[-1][cur_resolution].mean():0.3f}')
            std.append(F'{hists[-1][cur_resolution].std():0.3f}')
        print(F'{cur_resolution} AVG:{avg}     STD:{std} (train, validation, test)')

    print("Done!")
        # plotImages(final_img_folder, hists[1], cur_resolution,title, output_imgs, F'{cur_ctr}_{cur_dataset}__{cur_model}_{cur_resolution}')

def writeToFile(fileName, data, mode='w'):
    """ Writes 'data' to a file name. mode it can be 'a' or 'w' to appendo or write """
    f = open(fileName,mode)
    if mode == "a":
        f.write('\n')
    f.write(data)

def numToCases(arr):
    return [F'Case-{x:04d}' for x in arr]

def caseToNumber(arr):
    return [int(x.split('-')[1]) for x in arr if x.find('Case') != -1]

def plotHists(hists, title, labels):
    colors = ['b', 'g']
    for idx_col, data in enumerate(hists):
        plt.hist(data[cur_resolution].values, bins=np.arange(.2,.97,.01),color=colors[idx_col],label=labels[idx_col])
    plt.legend(loc='center left')
    plt.title(title)
    plt.show()

def getLatestFile(file_names):
    latest_file = ''
    largest_date = -1
    for cur_file in file_names:
        cur_time = getmtime(cur_file)
        if cur_time > largest_date:
            largest_date = cur_time
            latest_file = cur_file

    return latest_file

def getEarliestFile(file_names):
    latest_file = ''
    smallest_date = -1
    for cur_file in file_names:
        cur_time = getmtime(cur_file)
        if (cur_time <= smallest_date) or (smallest_date == -1):
            smallest_date = cur_time
            smallest_file = cur_file

    return smallest_file

def removeBlanks(line):
    return line.replace('   ', ' ').replace('  ', ' ').replace('\n', '').replace('Case-', '').replace('\'', '')

def appendValues(arr, values):
    for x in values:
        if x != '':
            arr.append(int(x))
    return arr

def findFile(folder,name):
    allFiles = listdir(folder)
    matched_files = [join(folder,x) for x in allFiles if not (re.search(name, x) is None)]
    if len(matched_files) > 0:
        # Here we should search for the latest one, TODO
        return getEarliestFile(matched_files)
    else:
        print(F"ERROR: didn't file the file I'm looking for!!! {folder} ---- {name}")

def readingSplitsFromModel(folder, model_names):
    '''
    Reads splits information from a txt file.
    :param folder:
    :param model_names:
    :param ver: It can be int or txt. int for splits with numbers and txt for string = 'Case....
    :return:
    '''
    print('**** Reading splits from the models ****')
    mydict = {}
    train_from_others = False  # This is used when the cases for the training are not written explicitly, but with '...'
    for ii, cur_model in enumerate(model_names):
        metrics_csv = findFile(folder,F'^{cur_model}')
        # print(F'---- Reading file {metrics_csv} ----')
        f = open(F'{metrics_csv}', "r")
        status = 'start'
        train = []
        val = []
        test = []
        for line in f:
            line = removeBlanks(line)
            # ------------ Reading training examples --------
            if (status == 'start') and (line.find('Train') != -1):
                if (line.find('...') != -1) or train_from_others:
                    train_from_others = True
                else:
                    values = line.split('[')[1].replace(']', '').split(' ')[0:]
                    train = appendValues(train, values)
                status = 'train'
                continue
            if status == 'train':
                if line.find(']') != -1: # Last line of train examples
                    if not(train_from_others):
                        values = line.split(']')[0].split(' ')[0:]
                        train = appendValues(train, values)
                    status = 'donetrain'
                    continue
                else: # Still reading train examples
                    values = line.replace('\n', '').split(' ')[0:]
                    train = appendValues(train, values)
                    continue
            # ------------ Reading validation examples --------
            if (status == 'donetrain') and (line.find('Validation') != -1):
                if line.find(']') == -1:
                    status = 'val'
                    values = line.split('[')[1].replace(']', '').split(' ')[0:]
                else:
                    status = 'doneval'
                    values = line.split('[')[1].split(']')[0].split(' ')[0:]
                val = appendValues(val, values)
                continue
            if status == 'val':
                if line.find(']') != -1: # Last line of train examples
                    values = line.split(']')[0].replace('\n', '').split(' ')[0:]
                    val = appendValues(val, values)
                    status = 'doneval'
                    continue
                else: # Still reading train examples
                    values = line.replace('\n', '').split(' ')[1:]
                    val = appendValues(val, values)
                    continue
            # ------------ Reading test examples --------
            if (status == 'doneval') and (line.find('Test') != -1):
                if line.find(']') == -1:
                    status = 'test'
                    values = line.split('[')[1].replace(']', '').split(' ')[0:]
                else:
                    status = 'donetest'
                    values = line.split('[')[1].split(']')[0].split(' ')[0:]
                test = appendValues(test, values)
                continue
            if status == 'test':
                if line.find(']') != -1: # Last line of train examples
                    values = line.split(']')[0].replace('\n', '').split(' ')[1:]
                    test = appendValues(test, values)
                    status = 'test'
                    continue
                else: # Still reading train examples
                    values = line.replace('\n', '').split(' ')[1:]
                    test = appendValues(test, values)
                    continue

        if train_from_others:
            latest_case = max([max(val), max(test)])
            val_np = np.array(val)
            test_np = np.array(test)
            all_cases = range(latest_case)
            train = np.setdiff1d(all_cases, [val_np, test_np])
            mydict[F'{cur_model}_train'] = train
        else:
            mydict[F'{cur_model}_train'] = train
        mydict[F'{cur_model}_validation'] = val
        mydict[F'{cur_model}_test'] = test
    return mydict

def getMeanCase(ser, val):
    diff = ser - val
    abs_diff = diff.abs()
    sort_diff = abs_diff.sort_values()
    return sort_diff.index[0]

def plotCurCase(case,folder_name, idx, cur_resolution):
    if cur_resolution == 'ROI':
        name_to_search = F'ROI\S*{case}'
    else:
        name_to_search = F'^{case}'
    metrics_csv = findFile(folder_name,name_to_search)
    cur_img = mpimg.imread(metrics_csv)
    if len(metrics_csv) > 0:
        plt.subplot(1,3,idx)
        plt.imshow(cur_img)
        plt.axis('off')
    return cur_img

def normTo255(img):
    data = 255 * img# Now scale by 255
    return data.astype(np.uint8)

def plotImages(folder_name, df, cur_resolution, title, output_folder, img_title):
    '''
    It computes the min, mean, and max values of the DSC and it searches the proper images inside the folder
    :param folder_name:
    :param df:
    :param cur_resolution:
    :return:
    '''
    allFiles = listdir(folder_name)

    minval = df[cur_resolution].min()
    meanval = df[cur_resolution].mean()
    maxval = df[cur_resolution].max()

    plt.figure(figsize=(24,8))
    mincase = df.index[df[cur_resolution] == minval].values[0]
    min_img = plotCurCase(mincase, folder_name,1, cur_resolution)
    imageio.imwrite(join(output_folder,F'{img_title}_MIN_{mincase}.png'), normTo255(min_img))
    meancase = getMeanCase(df[cur_resolution], meanval)
    mean_img = plotCurCase(meancase, folder_name,2, cur_resolution)
    imageio.imwrite(join(output_folder,F'{img_title}_MEAN_{meancase}.png'), normTo255(mean_img))
    # In order to show it in the middle it should be here
    plt.title(title, fontsize=20)
    maxcase = df.index[df[cur_resolution] == maxval].values[0]
    max_img = plotCurCase(maxcase, folder_name,3, cur_resolution)
    imageio.imwrite(join(output_folder,F'{img_title}_MAX_{maxcase}.png'), normTo255(max_img))
    print(F'{mincase} Min:{minval:0.3f} {meancase} Mean:{meanval:0.3f} {maxcase} Max:{maxval:0.3f}')
    plt.show()

if __name__ == "__main__":
    main()

# # ======================= FOR KIDNEYS DELETE ==================
#     exit()
#
#
#     for idx_model, cur_model in enumerate(model_names): # Iterate models
#         for idx_dataset, cur_dataset in enumerate(datasets): # Iterate databases (GE, ProstateX)
#             matched_folder = findFile(join(result_folder_csv,cur_dataset,cur_ctr),F'^{cur_model}')
#             # print(F'^^^^^ Reading data from: {matched_folder}')
#             # This is the file were we will search for DSC scores
#             final_img_folder = join(result_folder_csv,matched_folder)
#             metrics_csv = join(final_img_folder, 'all_DSC.csv')
#             # print(F'_____ Final file used {metrics_csv} ____')
#             df = pd.read_csv(join(cur_ctr,metrics_csv), index_col=0)
#             df.dropna(axis=0,how='any', inplace=True)
#             df = df.drop(numToCases(remove_cases[idx_dataset]),errors='ignore')
#             new_name = F'{cur_dataset}---{cur_model}.csv'
#             df.to_csv(join(output_files,new_name))
#
#             for idx_res, cur_resolution in enumerate(['ROI', 'Original']):
#                 print(F'******** {cur_model}-{cur_dataset}--{cur_resolution}******** ')
#                 # If we are plotting the model on the original dataset, then we need to filter the train vs validation
#                 if idx_model == idx_dataset:
#                     hists = []
#                     avg = []
#                     std = []
#                     for kk, cur_subset in enumerate(['train', 'validation']):
#                         indexes = numToCases(splits[F'{cur_model}_{cur_subset}'])
#                         # print(F'Filtering the cases for model {cur_model} and dataset:{cur_dataset}_{cur_subset}: tot_case: {len(indexes)}')
#                         hists.append(df.filter(items=indexes, axis=0))
#                         avg.append(F'{hists[-1][cur_resolution].mean():0.3f}')
#                         std.append(F'{hists[-1][cur_resolution].std():0.3f}')
#
#                     # Plot the images for the 'validation' set
#                     title = F'AVG:{avg}     STD:{std} (train, validation)  \n  Model:{cur_model}  Dataset:{cur_dataset}   Ctr:{cur_ctr}   Area:{cur_resolution}'
#                     plotImages(final_img_folder, hists[1], cur_resolution,title, output_imgs, F'{cur_ctr}_{cur_dataset}__{cur_model}_{cur_resolution}')
#                     title = F'AVG:{avg}     STD:{std} (train, validation) \n Model:{cur_model}  Dataset:{cur_dataset}   Ctr:{cur_ctr}   Area:{cur_resolution}'
#                     t1 = F'AVG:{avg} STD:{std}  {cur_model}-{cur_dataset} {cur_resolution}'
#                     summary_txt += t1 +'\n'
#                     print(t1)
#                     plotHists(hists,title,labels=['Training', 'Validation'])
#
#                 else:
#                     filt_df = df.filter(regex='Case*', axis=0) # We don't want the average (computed manually)
#                     avg = filt_df[cur_resolution].mean()
#                     std= filt_df[cur_resolution].std()
#                     title = F'AVG {avg:0.3f} STD {std:0.3f}   Model:{cur_model}  Dataset:{cur_dataset} \n Ctr:{cur_ctr} Subset:ALL  Area: {cur_resolution}'
#                     t2 = F'AVG:{avg:0.3f} STD:{std:0.3f}  {cur_model}-{cur_dataset} {cur_resolution}'
#                     summary_txt += t2 +'\n'
#                     print(t2)
#                     plotHists([filt_df],title,labels=['All'])
#                     # plt.bar(caseToNumber(filt_df.index.values), filt_df[cur_resolution].values)
#
#     writeToFile(join(output_files,F'{cur_ctr}_summary.txt'), summary_txt)
