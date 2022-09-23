# -------------------------------------------------------------------------
# 상기 프로그램에 대한 저작권을  포함한 지적재산권은 Deepnoid에 있으며,
# Deepnoid가 명시적으로 허용하지 않은 사용, 복사, 변경, 제3자에의 공개,
# 배포는 엄격히 금지되며, Deepnoid의 지적재산권 침해에 해당됩니다.
# (Copyright ⓒ 2020 Deepnoid Co., Ltd. All Rights Reserved|Confidential)
# -------------------------------------------------------------------------
# You are strictly prohibited to copy, disclose, distribute, modify,
# or use this program in part or as a whole without the prior written
# consent of Deepnoid Co., Ltd. Deepnoid Co., Ltd., owns the
# intellectual property rights in and to this program.
# (Copyright ⓒ 2020 Deepnoid Co., Ltd. All Rights Reserved|Confidential)
# -------------------------------------------------------------------------

import logging # Logs above the info level are exposed on the Log tab at the bottom of the module. (ex. logging.info('msg'))
from deepphi.image_processing import Preprocessing
import SimpleITK as sitk
from deepphi.io.sitk import *
import numpy as np
import cv2

class Resize(Preprocessing):
    def __init__(self, volume, height, width):
        super(Resize, self).__init__()
        self.z = volume
        self.y = height
        self.x = width
        

    def __call__(self, data, save_path=None):
        self.header_check(data)
        # new size calculate
        self.new_size = self.cal_size(data, [self.x, self.y, self.z])

        # Original Image
        img_array = data['image']['array']
        img_header = data['image']['header']
        self.img_info = [img_header['IsVector'], img_header['Direction'], img_header['Origin'], img_header['Spacing']]
        # image resize
        new_image = self.img_resize(img_array, self.img_info)
        # image save
        data['image'] = self.image_update(data['image'], new_image)
        data['image']['header']['dtype'] = str(data['image']['array'].dtype)
        data['image']['header']['original_shape'] = img_array.shape[0:3]

        # Segmentation Image
        self.label_image_resize(data, img_type='label', label_type='segmentation', array='array')
        self.label_image_resize(data, img_type='prediction', label_type='segmentation', array='array')  

        # Transformation Image
        self.label_image_resize(data, img_type='label', label_type='transformation', array='array')
        self.label_image_resize(data, img_type='prediction', label_type='transformation', array='array')

        # Detection Image
        self.detection_resize(data, img_type='label')
        self.detection_resize(data, img_type='prediction')        

        # Classification grad cam Image
        self.label_image_resize(data, img_type='prediction', label_type='classification', array='grad_cam')                       

        return data 
               
    def cal_size(self, data, tmp_size):
        img_dimension = data['image']['header']['dim']
        self.original_shape = data['image']['array'].shape
        return [self.original_shape[img_dimension - 1 - i] if tmp_size[i] == 'None' else tmp_size[i] for i in range(img_dimension)]

    def img_resize(self, img_array, img_info):
        sitk_image = sitk.GetImageFromArray(img_array, img_info[0])
        for i in range(len(img_info[1])):
            if img_info[1][i] == 0:
                img_info[1][i] = 0.00001
        sitk_image.SetDirection(img_info[1])
        sitk_image.SetOrigin(img_info[2])
        sitk_image.SetSpacing(img_info[3])

        new_spacing = [(ospc * osz / nsz) for osz, ospc, nsz in
                        zip(sitk_image.GetSize(), sitk_image.GetSpacing(), self.new_size)]

        sitk_image = sitk.Resample(sitk_image, self.new_size, sitk.Transform(),
                                            sitk.sitkLinear,
                                            sitk_image.GetOrigin(),
                                            new_spacing, sitk_image.GetDirection(),
                                            0, sitk_image.GetPixelID())
        return sitk_image

    def label_image_resize(self, data, label_type=None, img_type=None, array=None):
        if self.key_check(label_type, data[img_type], array):
            label_array = data[img_type][label_type][array]
            if img_type == 'label':
                label_header = data['label'][label_type]['header']
                self.label_info = [label_header['IsVector'], label_header['Direction'], label_header['Origin'], label_header['Spacing']]
            if label_type == 'classification':
                self.label_info = [True, self.img_info[1], self.img_info[2],self.img_info[3]]
            # label resize
            label_image = self.img_resize(label_array, self.label_info)
            # label save
            if img_type == 'label':
                data[img_type][label_type] = self.image_update(data[img_type][label_type], label_image)
            else:
                data[img_type][label_type][array] = sitk.GetArrayFromImage(label_image)
            if (label_type == 'segmentation') and (self.label_info[0] == True):
                label_array = self.segmentation_one_hot(data[img_type][label_type][array])
                data[img_type][label_type][array] = label_array

    def detection_resize(self, data, img_type=None):
        if self.key_check('object_detection', data[img_type], 'bbox_coordinate'):
            bbox_array = data[img_type]['object_detection']['bbox_coordinate']
            img_dimension = data['image']['header']['dim']
            for i in range(img_dimension):
                bbox_array[:,[i,i + img_dimension]] = (bbox_array[:,[i,i + img_dimension]]/self.original_shape[i])*self.new_size[img_dimension-1-i]
            bbox_new = np.trunc(bbox_array)
            data[img_type]['object_detection']['bbox_coordinate']= bbox_new

    def segmentation_one_hot(self, segmentation_array):
        ch_idx = np.argmax(segmentation_array, axis=-1)
        one_hot_img = np.zeros_like(segmentation_array)
        for i in range(segmentation_array.shape[-1]):
            one_hot_img[..., i] = (ch_idx == i).astype(int)
        return one_hot_img
    
    def key_check(self, key, dictionary, array):
        if key in dictionary:
            if array in dictionary[key]:
                if np.any(dictionary[key][array]):
                    return True
        else:
            return False

    # data check
    def header_check(self, data):
        ## feature merge check
        #if 'feature_merge' in data['image']['header']:
        #    raise ValueError("After 'Feature Merge' module, image processing module is not available. \n"
        #                     "Only Neural Network modules can be connected.")
        
        # dimension check
        if 'dim' not in data['image']['header']:
            raise KeyError("Dimension information is required to process. \nPlease add the Dimension information.")
        # Direction check
        img_direction = data['image']['header']['Direction']
        for i in range(len(img_direction)):
            if img_direction[i] == 0:
                img_direction[i] = 1e-6
        data['image']['header']['Direction'] = img_direction

    def image_update(self, data, sitk_image):
        """
        SimpleITK image object를 DeepPhiImage Object로 변환
        """
        array = sitk.GetArrayFromImage(sitk_image)
        spacing = list(sitk_image.GetSpacing())
        origin = list(sitk_image.GetOrigin())
        direction = list(sitk_image.GetDirection())

        ## squeeze ###
        if 1 in sitk_image.GetSize():
            array = array.squeeze()
            dim = sitk_image.GetDimension()
            index = sitk_image.GetSize().index(1)
            spacing = spacing[:index] + spacing[index + 1:]
            origin = origin[:index] + origin[index + 1:]
            direction = direction[:dim * index] + direction[dim * (index + 1):]

            i = 0
            new_direction = list()
            while direction:
                d = direction.pop(0)
                if i % dim != index:
                    new_direction.append(d)
                i += 1

            direction = new_direction

        data['array'] = array
        # image info
        data['header']['Direction'] = direction
        data['header']['Origin'] = origin
        data['header']['Spacing'] = spacing
        data['header']['IsVector'] = sitk_image.GetNumberOfComponentsPerPixel() > 1
        # array info
        ndim = len(data['array'].shape)
        if data['header']['IsVector']:
            dim = ndim - 1
        else:
            dim = ndim

        data['header']['dim'] = dim
        data['header']['ndim'] = ndim
        data['header']['dtype'] = str(data['array'].dtype)

        return data