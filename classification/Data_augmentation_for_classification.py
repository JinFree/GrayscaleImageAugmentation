import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def make_class_list(dataset_name = list_of_datas[0]):
    path_list = os.listdir(dataset_name)
    path_list.sort()
    class_list = []
    for idx in range(len(path_list)):
        path = os.path.join(dataset_name, path_list[idx])
        if os.path.isdir(path):
            class_list.append(path_list[idx])
    return class_list

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

        
def augmentation(original_data_path = list_of_datas[0]):
    path_1channel = original_data_path + "_1channel"
    
    if not os.path.isdir(path_1channel):
        os.mkdir(path_1channel)
    
    class_list = make_class_list(original_data_path)
    class_list.sort()
    
    original_images_list = []
    original_images_extension_list = []
    original_images_class_list = []
    src_class_path_list = []
    dst_class_path_list = []
    
    for idx in range(len(class_list)):
        src_class_path_list.append(os.path.join(original_data_path, class_list[idx]))
        dst_class_path_list.append(os.path.join(path_1channel, class_list[idx]))
        if not os.path.isdir(dst_class_path_list[idx]):
            os.mkdir(dst_class_path_list[idx])
    
    for class_idx in range(len(class_list)):
        original_class_path = src_class_path_list[class_idx]
        dst_class_path = dst_class_path_list[class_idx]
        image_list = os.listdir(original_class_path)
        for idx in tqdm(range(len(image_list))):
            src_image = os.path.join(original_class_path, image_list[idx])
            dst_image = os.path.join(dst_class_path, image_list[idx])
            dst_augmentation = dst_image[:-4]
            ret, cv_image = imageRead(src_image)
            if ret:
                original_images_list.append(image_list[idx][:-4])
                original_images_extension_list.append(image_list[idx][-4:])
                original_images_class_list.append(class_list[class_idx])
                
                if not os.path.isfile(dst_image):
                    shutil.copy(src_image, dst_image)
                    
                dst_augmentation_0 = dst_augmentation + augmentation_list[0] + dst_image[-4:]
                if not os.path.isfile(dst_augmentation_0):
                    res_image = random_pixel_noise(cv_image)
                    cv2.imwrite(dst_augmentation_0, res_image)
                    
                dst_augmentation_1 = dst_augmentation + augmentation_list[1] + dst_image[-4:]
                if not os.path.isfile(dst_augmentation_1):
                    res_image = CLAHE(cv_image)
                    cv2.imwrite(dst_augmentation_1, res_image)
                    
                dst_augmentation_2 = dst_augmentation + augmentation_list[2] + dst_image[-4:]
                if not os.path.isfile(dst_augmentation_2):
                    res_image = gaussian_blur(cv_image)
                    cv2.imwrite(dst_augmentation_2, res_image)
                    
                dst_augmentation_3 = dst_augmentation + augmentation_list[3] + dst_image[-4:]
                if not os.path.isfile(dst_augmentation_3):
                    res_image = morphological_edge(cv_image)
                    cv2.imwrite(dst_augmentation_3, res_image)
                    
    return class_list, original_images_list, original_images_extension_list, original_images_class_list

def augmentation_3(original_data_path = list_of_datas[0]
                   , class_list = None
                   , original_images_list = None
                   , original_images_extension_list=None
                   , original_images_class_list = None):
    if original_images_list is None:
        return
    
    path_1channel = original_data_path + "_1channel"
    path_3channel_color = original_data_path + "_3channel_color"
    path_3channel_gray = original_data_path + "_3channel_gray"
    
    if not os.path.isdir(path_3channel_color):
        os.mkdir(path_3channel_color)
        
    if not os.path.isdir(path_3channel_gray):
        os.mkdir(path_3channel_gray)
    
    for idx in range(len(class_list)):
        dst_color_class = os.path.join(path_3channel_color, class_list[idx])
        if not os.path.isdir(dst_color_class):
            os.mkdir(dst_color_class)
            
        dst_color_gray = os.path.join(path_3channel_gray, class_list[idx])
        if not os.path.isdir(dst_color_gray):
            os.mkdir(dst_color_gray)
    
    for idx in tqdm(range(len(original_images_list))):
        src_path = os.path.join(path_1channel, original_images_class_list[idx], original_images_list[idx])
        img_extension = original_images_extension_list[idx]
        
        dst_color_path = os.path.join(path_3channel_color, original_images_class_list[idx], original_images_list[idx])
        dst_gray_path = os.path.join(path_3channel_gray, original_images_class_list[idx], original_images_list[idx])
        
        original_path = src_path + img_extension
        PN_PATH = src_path + augmentation_list[0] + img_extension
        CL_PATH = src_path + augmentation_list[1] + img_extension
        GB_PATH = src_path + augmentation_list[2] + img_extension
        ME_PATH = src_path + augmentation_list[3] + img_extension
        

        IMAGE_OR = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        IMAGE_PN = cv2.imread(PN_PATH, cv2.IMREAD_GRAYSCALE)
        IMAGE_CL= cv2.imread(CL_PATH, cv2.IMREAD_GRAYSCALE)
        IMAGE_GB = cv2.imread(GB_PATH, cv2.IMREAD_GRAYSCALE)
        IMAGE_ME = cv2.imread(ME_PATH, cv2.IMREAD_GRAYSCALE)
        
        '''
        case0 = OR, ME, PN
        case1 = OR, ME, GB
        case2 = OR, ME, CL
        case3 = PN, ME, GB
        case4 = PN, ME, CL
        case5 = GB, ME, CL
        '''
        
        dst_path = dst_color_path + img_extension
        
        if not os.path.isfile(dst_path):
            shutil.copy(original_path, dst_path)
        
        AUG_OMP = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_PN)
        AUG_OMG = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_GB)
        AUG_OMC = mergeImage(IMAGE_OR, IMAGE_ME, IMAGE_CL)
        AUG_PMG = mergeImage(IMAGE_PN, IMAGE_ME, IMAGE_GB)
        AUG_PMC = mergeImage(IMAGE_PN, IMAGE_ME, IMAGE_CL)
        AUG_GMC = mergeImage(IMAGE_GB, IMAGE_ME, IMAGE_CL)
        
        dst_path = dst_color_path + augmentation_list_2nd[0] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMP)
        
        dst_path = dst_color_path + augmentation_list_2nd[1] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMG)
            
        dst_path = dst_color_path + augmentation_list_2nd[2] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMC)
            
        dst_path = dst_color_path + augmentation_list_2nd[3] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_PMG)
            
        dst_path = dst_color_path + augmentation_list_2nd[4] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_PMC)
            
        dst_path = dst_color_path + augmentation_list_2nd[5] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_GMC)
            
        AUG_OMP = cv2.cvtColor(AUG_OMP, cv2.COLOR_BGR2GRAY)
        AUG_OMC = cv2.cvtColor(AUG_OMC, cv2.COLOR_BGR2GRAY)
        AUG_OMG = cv2.cvtColor(AUG_OMG, cv2.COLOR_BGR2GRAY)
        AUG_PMC = cv2.cvtColor(AUG_PMC, cv2.COLOR_BGR2GRAY)
        AUG_PMG = cv2.cvtColor(AUG_PMG, cv2.COLOR_BGR2GRAY)
        AUG_GMC = cv2.cvtColor(AUG_GMC, cv2.COLOR_BGR2GRAY)    
        
        dst_path = dst_gray_path + img_extension
        
        if not os.path.isfile(dst_path):
            shutil.copy(original_path, dst_path)    
        
        dst_path = dst_gray_path + augmentation_list_2nd[0] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMP)
        
        dst_path = dst_gray_path + augmentation_list_2nd[1] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMG)
            
        dst_path = dst_gray_path + augmentation_list_2nd[2] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_OMC)
            
        dst_path = dst_gray_path + augmentation_list_2nd[3] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_PMG)
            
        dst_path = dst_gray_path + augmentation_list_2nd[4] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_PMC)
            
        dst_path = dst_gray_path + augmentation_list_2nd[5] + img_extension        
        if not os.path.isfile(dst_path):
            cv2.imwrite(dst_path, AUG_GMC)
        
                
if __name__=="__main__":
    for data in list_of_datas:
        print("do 1 channel augmentation")
        class_list, original_images_list, original_images_extension_list, original_images_class_list = augmentation(data)
        print("do 3 channel augmentation")
        augmentation_3(data, class_list, original_images_list, original_images_extension_list, original_images_class_list)