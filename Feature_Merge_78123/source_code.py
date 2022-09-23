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
import numpy as np
import cv2
from deepphi.image_processing import Preprocessing
from deepphi.logger.error import *


class FeatureMerge(Preprocessing):
    def __init__(self):
        super(FeatureMerge, self).__init__()

    def __call__(self, *data, save_path=None):
        data_list = data # input data list
        new_data = data_list[0] 
        image_dimension = new_data['image']['header']['dim']
        image_array = np.float32(np.copy(new_data['image']['array']))
        image_color = [new_data['image']['header']['color_mode']]
        add_image_color = []
        # main merge algorithm
        for i in range(1,len(data_list)):
            new_dimension = data_list[i]['image']['header']['dim']
            new_array = np.float32(data_list[i]['image']['array'])
            new_color = data_list[i]['image']['header']['color_mode']
            if image_dimension != new_dimension:
                code = 'image-processing.error.dimension-equal'
                msg = "The dimension of datas are not the same. Please make the dimension the same."
                raise DeepPhiError(code=code, msg=msg, parameter={})

            if image_dimension == 2:
                if (image_array.shape[0:2] != new_array.shape[0:2]):
                    code = 'image-processing.error.shape-equal'
                    msg = "The height and width of Data1 and Data%d are not the same. Please make the height and width the same." %(i+1)
                    raise DeepPhiError(code=code, msg=msg, parameter={'shape': 'height and width', 'number': '%d' %(i+1)})

                image_array = cv2.merge([image_array,new_array])
            if image_dimension == 3:
                if len(image_array.shape) == 3:
                    image_array = np.expand_dims(image_array,-1)
                if len(new_array.shape) == 3:
                    new_array = np.expand_dims(new_array,-1)
                if (image_array.shape[0:3] != new_array.shape[0:3]):
                    code = 'image-processing.error.shape-equal'
                    msg = "The height, width and volumn of Data1 and Data%d are not the same. Please make the height, width and volumn the same." %(i+1)
                    raise DeepPhiError(code=code, msg=msg, parameter={'shape': 'height, width and volumn', 'number': '%d' %(i+1)})

                new_3d_array = np.float32(np.zeros((image_array.shape[0],image_array.shape[1],image_array.shape[2],image_array.shape[3]+new_array.shape[3])))
                if image_array.shape[3] == 1:
                    image_array = np.squeeze(image_array,-1)
                if new_array.shape[3] == 1:
                    new_array = np.squeeze(new_array,-1)
                for j in range(image_array.shape[0]):
                    new_3d_array[j,] = cv2.merge([image_array[j,],new_array[j,]])
                image_array = new_3d_array
            if new_color not in image_color:
                add_image_color.append(new_color)
        new_data['image']['array'] = image_array # output image
        if len(data_list) != 1:
            new_data['image']['header']['IsVector'] = True
        new_data['image']['header']['added_color_mode'] = add_image_color # color mode change
        # new_data['image']['header']['feature_merge'] = True
        new_data['image']['header']['color_mode'] = 'Series'
        new_data['image']['header']['dtype'] = 'float32' # data type change
        return new_data
