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
import cv2, copy
from deepphi.image_processing import Preprocessing
from deepphi.logger.error import *
from deepphi.io.sitk import *
import numpy as np


class ZeroPadding(Preprocessing):
    def __init__(self):
        super(ZeroPadding, self).__init__()

    def __call__(self, data):
        # input image
        self.header_check(data)

        # Original Image
        img_array = data['image']['array']
        img_header = data['image']['header']
        new_array, margin_list = self.apply_margin(img_array, img_header)

        data['image']['header']['padding'] = margin_list
        data['image']['array'] = new_array # output image
        data['image']['header']['dtype'] = str(new_array.dtype)

        # Segmentation Image
        self.label_image_padding(data, img_type='label', label_type='segmentation', array='array')
        self.label_image_padding(data, img_type='prediction', label_type='segmentation', array='array')

        # Transformaion Image
        self.label_image_padding(data, img_type='label', label_type='transformation', array='array')
        self.label_image_padding(data, img_type='prediction', label_type='transformation', array='array')

        # Grad_cam Image
        self.label_image_padding(data, img_type='prediction', label_type='classification', array='grad_cam')
        
        # Detection Image
        self.detection_padding(data, img_array.shape, img_header['dim'], 'label')
        self.detection_padding(data, img_array.shape, img_header['dim'], 'prediction')

        return data    

    def make_margin(self, img_shape, img_header):
        dimension = img_header['dim']
        height, width = img_shape[dimension-2:dimension]
        margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]
        if np.abs(height-width)%2 != 0:
            margin[0] += 1
        if height < width:
            margin_list = [margin, [0, 0]]
        else:
            margin_list = [[0, 0], margin]
        if img_header['IsVector'] == True:
            margin_list.append([0,0])
        if dimension == 3:
            margin_list.insert(0,[0,0])
        return margin_list
    
    def apply_margin(self, img_array, img_header, label_type=None):
        margin_list = self.make_margin(img_array.shape, img_header)        
        new_array = np.pad(img_array, margin_list, mode='constant') # add zero-padding to image
        if label_type == 'segmentation':
            new_array[..., 0] = np.pad(img_array[..., 0], margin_list[: -1], mode='constant', constant_values=1)
        return new_array, margin_list

    def label_image_padding(self, data, img_type=None, label_type=None, array=None):
        if self.key_check(label_type, data[img_type], array):
            label_array = data[img_type][label_type][array]
            if img_type == 'label':
                self.label_header = data[img_type][label_type]['header']
            if label_type == 'classification':
                self.label_header = copy.deepcopy(data['image']['header'])
                self.label_header['IsVector'] = True
            label_array, label_margin_list = self.apply_margin(label_array, self.label_header, label_type)
            data[img_type][label_type][array] = label_array
            if img_type == 'label':
                data[img_type][label_type]['header']['padding'] = label_margin_list

    def detection_padding(self, data, img_shape, dimension, img_type):
        if self.key_check('object_detection', data[img_type], 'bbox_coordinate'):
            label_array = data[img_type]['object_detection']['bbox_coordinate']
            new_array = np.copy(label_array)
            height, width = img_shape[dimension-2:dimension]
            margin = np.abs(height - width) // 2
            if np.abs(height-width)%2 != 0:
                margin += 1
            if height > width:
                new_array[:, [dimension-1, 2*dimension-1]] = (label_array[:, [dimension-1, 2*dimension-1]] + margin)
            else:
                new_array[:, [dimension-2, 2*dimension-2]] = (label_array[:, [dimension-2, 2*dimension-2]] + margin)
            data[img_type]['object_detection']['bbox_coordinate'] = new_array 
            if img_type == 'label':
                data['label']['object_detection']['header']['padding'] = [abs(label_array[0][dimension-2] - new_array[0][dimension-2]), abs(label_array[0][dimension-1] - new_array[0][dimension-1])]
    
    def key_check(self, key, dictionary, array):
        if key in dictionary:
            if array in dictionary[key]:
                if np.any(dictionary[key][array]):
                    return True
        else:
            return False

    # header check
    def header_check(self, data):
        # feature merge check
        if 'feature_merge' in data['image']['header']:
            code = 'image-processing.error.post-feature-merge'
            msg = "After 'Feature Merge' module, image processing module is not available."
            msg = msg + "Only Neural Network modules can be connected."
            raise DeepPhiError(code=code, msg=msg, parameter={})

        # dimension check
        if 'dim' not in data['image']['header']:
            code = 'image-processing.error.empty-header'
            msg = "Dimension information is required to process. Please add the Dimension information to the header."
            raise DeepPhiError(code=code, msg=msg, parameter={'header_key':'Dimension'})