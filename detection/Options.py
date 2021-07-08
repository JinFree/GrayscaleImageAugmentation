# EDIT LIST_OF_DATAS FOR YOUR USECASE
list_of_datas = ['train', 'validation']

''''
ex)
FROM

$(ROOT_TO_DATASET)/
                   train/
                         img1.jpg
                         img1.txt
                   validation/
                              img1.jpg
                              img1.txt
                   Optins.py
                   Data_augmentation_for_detection.py
                   
                   
TO

$(ROOT_TO_DATASET)/
                   train/
                         img1.jpg
                         img1.txt
                         
                         img1.CLAHE.jpg
                         img1.CLAHE.txt
                         img1.GAUSSIAN_BLUR.jpg
                         img1.GAUSSIAN_BLUR.txt
                         img1.MORPHOLOGICAL_EDGE.jpg
                         img1.MORPHOLOGICAL_EDGE.txt
                         img1.PIXEL_NOISE.jpg
                         img1.PIXEL_NOISE.txt
                         img1.CLAHE.jpg
                         img1.CLAHE.txt
                         img1.GAUSSIAN_BLUR.jpg
                         img1.GAUSSIAN_BLUR.txt
                         img1.MORPHOLOGICAL_EDGE.jpg
                         img1.MORPHOLOGICAL_EDGE.txt
                         img1.PIXEL_NOISE.jpg
                         img1.PIXEL_NOISE.txt
                         
                         img1.OR_ME_PN_COLOR.jpg
                         img1.OR_ME_PN_COLOR.txt
                         img1.OR_ME_GB_COLOR.jpg
                         img1.OR_ME_GB_COLOR.txt
                         img1.OR_ME_CL_COLOR.jpg
                         img1.OR_ME_CL_COLOR.txt
                         img1.PN_ME_GB_COLOR.jpg
                         img1.PN_ME_GB_COLOR.txt
                         img1.PN_ME_CL_COLOR.jpg
                         img1.PN_ME_CL_COLOR.txt
                         img1.GB_ME_CL_COLOR.jpg
                         img1.GB_ME_CL_COLOR.txt
                         
                         img1.OR_ME_PN_GRAY.jpg
                         img1.OR_ME_PN_GRAY.txt
                         img1.OR_ME_GB_GRAY.jpg
                         img1.OR_ME_GB_GRAY.txt
                         img1.OR_ME_CL_GRAY.jpg
                         img1.OR_ME_CL_GRAY.txt
                         img1.PN_ME_GB_GRAY.jpg
                         img1.PN_ME_GB_GRAY.txt
                         img1.PN_ME_CL_GRAY.jpg
                         img1.PN_ME_CL_GRAY.txt
                         img1.GB_ME_CL_GRAY.jpg
                         img1.GB_ME_CL_GRAY.txt
                         
                   validation/
                              img1.jpg
                              img1.txt
                         
                              img1.CLAHE.jpg
                              img1.CLAHE.txt
                              img1.GAUSSIAN_BLUR.jpg
                              img1.GAUSSIAN_BLUR.txt
                              img1.MORPHOLOGICAL_EDGE.jpg
                              img1.MORPHOLOGICAL_EDGE.txt
                              img1.PIXEL_NOISE.jpg
                              img1.PIXEL_NOISE.txt
                              img1.CLAHE.jpg
                              img1.CLAHE.txt
                              img1.GAUSSIAN_BLUR.jpg
                              img1.GAUSSIAN_BLUR.txt
                              img1.MORPHOLOGICAL_EDGE.jpg
                              img1.MORPHOLOGICAL_EDGE.txt
                              img1.PIXEL_NOISE.jpg
                              img1.PIXEL_NOISE.txt
                         
                              img1.OR_ME_PN_COLOR.jpg
                              img1.OR_ME_PN_COLOR.txt
                              img1.OR_ME_GB_COLOR.jpg
                              img1.OR_ME_GB_COLOR.txt
                              img1.OR_ME_CL_COLOR.jpg
                              img1.OR_ME_CL_COLOR.txt
                              img1.PN_ME_GB_COLOR.jpg
                              img1.PN_ME_GB_COLOR.txt
                              img1.PN_ME_CL_COLOR.jpg
                              img1.PN_ME_CL_COLOR.txt
                              img1.GB_ME_CL_COLOR.jpg
                              img1.GB_ME_CL_COLOR.txt
                              
                              img1.OR_ME_PN_GRAY.jpg
                              img1.OR_ME_PN_GRAY.txt
                              img1.OR_ME_GB_GRAY.jpg
                              img1.OR_ME_GB_GRAY.txt
                              img1.OR_ME_CL_GRAY.jpg
                              img1.OR_ME_CL_GRAY.txt
                              img1.PN_ME_GB_GRAY.jpg
                              img1.PN_ME_GB_GRAY.txt
                              img1.PN_ME_CL_GRAY.jpg
                              img1.PN_ME_CL_GRAY.txt
                              img1.GB_ME_CL_GRAY.jpg
                              img1.GB_ME_CL_GRAY.txt
                   Optins.py
                   Data_augmentation_for_detection.py
                   train_original.txt
                   validation_original.txt
                   train_1channel.txt
                   validation_1channel.txt
                   train_3channel_gray.txt
                   validation_3channel_gray.txt
                   train_3channel_color.txt
                   validation_3channel_color.txt
'''