import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from Options import *


augmentation_list = ['.PIXEL_NOISE', \
                             '.CLAHE', \
                             '.GAUSSIAN_BLUR', \
                             '.MORPHOLOGICAL_EDGE']

augmentation_list_2nd = ['.OR_ME_PN', \
                         '.OR_ME_GB', \
                         '.OR_ME_CL', \
                         '.PN_ME_GB', \
                         '.PN_ME_CL', \
                         '.GB_ME_CL']

dataset_form_list = ['1channel', '3channel_color', '3channel_gray']


def make_path_lists(dataset_name = list_of_datas[0]):
    path_list = os.listdir(dataset_name)
    path_list.sort()
    return_path_list = []
    for idx in range(len(path_list)):
        if path_list[idx] == ".ipynb_checkpoints":
            continue
        elif path_list[idx][-3:] == "txt":
            continue
        elif ("." + path_list[idx].split('.')[-2]) in augmentation_list:
            continue
        elif ("." + path_list[idx].split('.')[-2] + "_COLOR") in augmentation_list_2nd:
            continue
        elif ("." + path_list[idx].split('.')[-2] + "_GRAY") in augmentation_list_2nd:
            continue
        return_path_list.append(path_list[idx])
    return_path_list.sort()
    return return_path_list

def imageRead(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, None
    else:
        return True, image

def splitImage(image):
    return cv2.split(image)

def mergeImage(channel1, channel2, channel3):
    return cv2.merge((channel1, channel2, channel3))

def random_pixel_noise(cv_image, thresh = 10):
    height, width = cv_image.shape[0], cv_image.shape[1]
    for x in range(width):
        for y in range(height):
            value = cv_image[y, x]
            value += random.randint(-1 * thresh, thresh)
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            cv_image[y, x] = value
    return cv_image

def morphological_edge(cv_image, flag=cv2.MORPH_ELLIPSE , size=11, iterations=1):
    kernel = cv2.getStructuringElement(flag, (size, size))
    return cv2.morphologyEx(cv_image, op=cv2.MORPH_GRADIENT, kernel=kernel, iterations=iterations)

def CLAHE(cv_image, clipLimit_ = 2.0, tileGridSize_=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit_, tileGridSize=tileGridSize_)
    return clahe.apply(cv_image)

def gaussian_blur(cv_image, size=(11, 11), sigmaX=11, sigmaY=11):
    return cv2.GaussianBlur(cv_image, ksize=size, sigmaX=sigmaX, sigmaY=sigmaY)

def augmentation(folder_path, image_list):

    for idx in tqdm(range(len(image_list))):
        img_full_path = os.path.join(folder_path, image_list[idx])
        
        ret, cv_image = imageRead(img_full_path)
        
        
        if ret:
            file_name = image_list[idx][:-4]
            origin_txt = os.path.join(folder_path,file_name + ".txt")
            aug_0_img = os.path.join(folder_path,file_name + augmentation_list[0] + ".png")
            aug_0_txt = os.path.join(folder_path,file_name + augmentation_list[0] + ".txt")
            
            aug_1_img = os.path.join(folder_path,file_name + augmentation_list[1] + ".png")
            aug_1_txt = os.path.join(folder_path,file_name + augmentation_list[1] + ".txt")
            
            aug_2_img = os.path.join(folder_path,file_name + augmentation_list[2] + ".png")
            aug_2_txt = os.path.join(folder_path,file_name + augmentation_list[2] + ".txt")
            
            aug_3_img = os.path.join(folder_path,file_name + augmentation_list[3] + ".png")
            aug_3_txt = os.path.join(folder_path,file_name + augmentation_list[3] + ".txt")
            if not os.path.isfile(aug_0_img):
                res_image = random_pixel_noise(cv_image)
                cv2.imwrite(aug_0_img, res_image)
                shutil.copyfile(origin_txt, aug_0_txt)
                
            if not os.path.isfile(aug_1_img):
                res_image = CLAHE(cv_image)
                cv2.imwrite(aug_1_img, res_image)
                shutil.copyfile(origin_txt, aug_1_txt)
                
            if not os.path.isfile(aug_2_img):
                res_image = gaussian_blur(cv_image)
                cv2.imwrite(aug_2_img, res_image)
                shutil.copyfile(origin_txt, aug_2_txt)
                
            if not os.path.isfile(aug_3_img):
                res_image = morphological_edge(cv_image)
                cv2.imwrite(aug_3_img, res_image)
                shutil.copyfile(origin_txt, aug_3_txt)

def augmentation_2nd(folder_path, image_list):
    for idx in tqdm(range(len(image_list))):
        img_full_path = os.path.join(folder_path, image_list[idx])
        ret, IMAGE_OR = imageRead(img_full_path)
        if ret:
            file_name = image_list[idx][:-4]
            PN_PATH = os.path.join(folder_path,file_name + augmentation_list[0] + ".png")
            CLAHE_PATH = os.path.join(folder_path,file_name + augmentation_list[1] + ".png")
            GB_PATH = os.path.join(folder_path,file_name + augmentation_list[2] + ".png")
            ME_PATH = os.path.join(folder_path,file_name + augmentation_list[3] + ".png")
            
            IMAGE_PN = cv2.imread(PN_PATH, cv2.IMREAD_GRAYSCALE)
            IMAGE_CL= cv2.imread(CLAHE_PATH, cv2.IMREAD_GRAYSCALE)
            IMAGE_GB = cv2.imread(GB_PATH, cv2.IMREAD_GRAYSCALE)
            IMAGE_ME = cv2.imread(ME_PATH, cv2.IMREAD_GRAYSCALE)
            
            AUG_OMP = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_PN)
            AUG_OMG = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_GB)
            AUG_OMC = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_CL)
            AUG_PMG = mergeImage(IMAGE_PN, IMAGE_ME, IMAGE_GB)
            AUG_PMC = mergeImage(IMAGE_PN, IMAGE_ME, IMAGE_CL)
            AUG_GMC = mergeImage(IMAGE_GB, IMAGE_ME, IMAGE_CL)
            
            origin_txt = os.path.join(folder_path,file_name + ".txt")
            aug_0_img = os.path.join(folder_path,file_name + augmentation_list_2nd[0] + "_COLOR.png")
            aug_0_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[0] + "_COLOR.txt")
            
            aug_1_img = os.path.join(folder_path,file_name + augmentation_list_2nd[1] + "_COLOR.png")
            aug_1_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[1] + "_COLOR.txt")
            
            aug_2_img = os.path.join(folder_path,file_name + augmentation_list_2nd[2] + "_COLOR.png")
            aug_2_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[2] + "_COLOR.txt")
            
            aug_3_img = os.path.join(folder_path,file_name + augmentation_list_2nd[3] + "_COLOR.png")
            aug_3_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[3] + "_COLOR.txt")
            
            aug_4_img = os.path.join(folder_path,file_name + augmentation_list_2nd[4] + "_COLOR.png")
            aug_4_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[4] + "_COLOR.txt")
            
            aug_5_img = os.path.join(folder_path,file_name + augmentation_list_2nd[5] + "_COLOR.png")
            aug_5_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[5] + "_COLOR.txt")
            if not os.path.isfile(aug_0_img):
                cv2.imwrite(aug_0_img, AUG_OMP)
                shutil.copyfile(origin_txt, aug_0_txt)
                
            if not os.path.isfile(aug_1_img):
                cv2.imwrite(aug_1_img, AUG_OMG)
                shutil.copyfile(origin_txt, aug_1_txt)
                
            if not os.path.isfile(aug_2_img):
                cv2.imwrite(aug_2_img, AUG_OMC)
                shutil.copyfile(origin_txt, aug_2_txt)
                
            if not os.path.isfile(aug_3_img):
                cv2.imwrite(aug_3_img, AUG_PMG)
                shutil.copyfile(origin_txt, aug_3_txt)
                
            if not os.path.isfile(aug_4_img):
                cv2.imwrite(aug_4_img, AUG_PMC)
                shutil.copyfile(origin_txt, aug_4_txt)
                
            if not os.path.isfile(aug_5_img):
                cv2.imwrite(aug_5_img, AUG_GMC)
                shutil.copyfile(origin_txt, aug_5_txt)
                
            AUG_OMP = cv2.cvtColor(AUG_OMP, cv2.COLOR_BGR2GRAY)
            AUG_OMC = cv2.cvtColor(AUG_OMC, cv2.COLOR_BGR2GRAY)
            AUG_OMG = cv2.cvtColor(AUG_OMG, cv2.COLOR_BGR2GRAY)
            AUG_PMG = cv2.cvtColor(AUG_PMG, cv2.COLOR_BGR2GRAY)
            AUG_PMC = cv2.cvtColor(AUG_PMC, cv2.COLOR_BGR2GRAY)
            AUG_GMC = cv2.cvtColor(AUG_GMC, cv2.COLOR_BGR2GRAY)
            
            aug_0_img = os.path.join(folder_path,file_name + augmentation_list_2nd[0] + "_GRAY.png")
            aug_0_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[0] + "_GRAY.txt")
            
            aug_1_img = os.path.join(folder_path,file_name + augmentation_list_2nd[1] + "_GRAY.png")
            aug_1_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[1] + "_GRAY.txt")
            
            aug_2_img = os.path.join(folder_path,file_name + augmentation_list_2nd[2] + "_GRAY.png")
            aug_2_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[2] + "_GRAY.txt")
            
            aug_3_img = os.path.join(folder_path,file_name + augmentation_list_2nd[3] + "_GRAY.png")
            aug_3_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[3] + "_GRAY.txt")
            
            aug_4_img = os.path.join(folder_path,file_name + augmentation_list_2nd[4] + "_GRAY.png")
            aug_4_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[4] + "_GRAY.txt")
            
            aug_5_img = os.path.join(folder_path,file_name + augmentation_list_2nd[5] + "_GRAY.png")
            aug_5_txt = os.path.join(folder_path,file_name + augmentation_list_2nd[5] + "_GRAY.txt")
            if not os.path.isfile(aug_0_img):
                cv2.imwrite(aug_0_img, AUG_OMP)
                shutil.copyfile(origin_txt, aug_0_txt)
                
            if not os.path.isfile(aug_1_img):
                cv2.imwrite(aug_1_img, AUG_OMG)
                shutil.copyfile(origin_txt, aug_1_txt)
                
            if not os.path.isfile(aug_2_img):
                cv2.imwrite(aug_2_img, AUG_OMC)
                shutil.copyfile(origin_txt, aug_2_txt)
                
            if not os.path.isfile(aug_3_img):
                cv2.imwrite(aug_3_img, AUG_PMG)
                shutil.copyfile(origin_txt, aug_3_txt)
                
            if not os.path.isfile(aug_4_img):
                cv2.imwrite(aug_4_img, AUG_PMC)
                shutil.copyfile(origin_txt, aug_4_txt)
                
            if not os.path.isfile(aug_5_img):
                cv2.imwrite(aug_5_img, AUG_GMC)
                shutil.copyfile(origin_txt, aug_5_txt)

                
def make_dataset_list_v2(dataset_name = list_of_datas[0], dataset_form = None ):
    path_list = os.listdir(dataset_name)
    path_list.sort()
    return_path_list = []
    for idx in range(len(path_list)):
        splitter_string = path_list[idx].split('.')[-2]
        if path_list[idx] == ".ipynb_checkpoints":
            continue
        elif path_list[idx][-3:] == "txt":
            continue
        elif ("." + splitter_string) in augmentation_list and (dataset_form is not dataset_form_list[0]):
            continue
        elif ("." + splitter_string[:-6]) in augmentation_list_2nd and (dataset_form is not dataset_form_list[1]):
            continue
        elif ("." + splitter_string[:-5]) in augmentation_list_2nd and (dataset_form is not dataset_form_list[2]):
            continue
        return_path_list.append(path_list[idx])
    return_path_list.sort()
    return return_path_list


def make_dataset_list_v3(dataset_name = list_of_datas[0]):
    path_list = os.listdir(dataset_name)
    path_list.sort()
    return_path_list = []
    for idx in range(len(path_list)):
        splitter_string = path_list[idx].split('.')[-2]
        filename_string = path_list[idx].split('.')[0]
        if path_list[idx].split('.')[-1] == 'txt':
            continue
        if splitter_string == filename_string:
            return_path_list.append(path_list[idx])
    return_path_list.sort()
    return return_path_list

if __name__=="__main__":
    for path_to_dataset in list_of_datas:
        image_lists = make_path_lists(path_to_dataset)
        augmentation(path_to_dataset, image_lists)
        augmentation_2nd(path_to_dataset, image_lists)
        for dataset_form in dataset_form_list:
            image_lists = make_dataset_list_v2(path_to_dataset, dataset_form)
            textFileName = path_to_dataset + "_" + dataset_form + ".txt"
            fid = open(textFileName, 'w')
            for file in image_lists:
                filepath = os.path.join(os.getcwd(), path_to_dataset, file)
                fid.write(filepath + '\n')
            fid.close()
        image_lists = make_dataset_list_v3(path_to_dataset)
        textFileName = path_to_dataset + "_" + "original" + ".txt"
        fid = open(textFileName, 'w')
        for file in image_lists:
            filepath = os.path.join(os.getcwd(), path_to_dataset, file)
            fid.write(filepath + '\n')
        fid.close()
