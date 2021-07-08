# EDIT LIST_OF_DATAS FOR YOUR USECASE
list_of_datas = ['train', 'validation']

''''
ex)
FROM

$(ROOT_TO_DATASET)/
                   train/
                         class1/
                                any images can open by opencv
                         class2/
                                images
                         class3/
                                images
                   validation/
                              class1/
                                     images
                              class2/
                                     images
                              class3/
                                     images
                   Optins.py
                   Data_augmentation_for_classification.py
                   
                   
TO

$(ROOT_TO_DATASET)/
                   train/
                         class1/
                                any images can open by opencv
                         class2/
                                images
                   validation/
                              class1/
                                     images
                              class2/
                                     images
                   train_1channel/
                                  1 channel augmented images
                   validation_1channel/
                                       1 channel augmented images
                   train_3channel_gray/
                                       3 channel augmented grayscale images
                   validation_3channel_gray/
                                            3 channel augmented grayscale images
                   train_3channel_color/
                                        3 channel augmented color images
                   validation_3channel_color/
                                             3 channel augmented color images
                   Optins.py
                   Data_augmentation_for_classification.py
'''