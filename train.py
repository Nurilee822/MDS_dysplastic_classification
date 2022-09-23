
import sys
sys.path.append("./library/")


import copy

from deepphi.io.sitk import DeepPhiDataSet
import Zero_padding_77958.source_code
import Resize_77959.source_code
import InceptionV3_77960.source_code
import InceptionV3_0_77985.source_code
import InceptionV3_1_78066.source_code
import Feature_Merge_78123.source_code


def inference(path_image):
    input_image = DeepPhiDataSet()
    input_image.read_image(path_image)

    output_Zero_padding_77958 = Zero_padding_77958.source_code.ZeroPadding()(input_image)
    output_Resize_77959 = Resize_77959.source_code.Resize(volume=None, width=64, height=64)(output_Zero_padding_77958)
    output_InceptionV3_77960 = InceptionV3_77960.source_code.Model(label_type='Classification 2D')(output_Resize_77959)
    output_InceptionV3_0_77985 = InceptionV3_0_77985.source_code.Model(label_type='Classification 2D')(output_Resize_77959)
    output_InceptionV3_1_78066 = InceptionV3_1_78066.source_code.Model(label_type='Classification 2D')(output_Resize_77959)
    output_Feature_Merge_78123 = Feature_Merge_78123.source_code.FeatureMerge()(output_InceptionV3_77960, output_InceptionV3_0_77985, output_InceptionV3_1_78066)

    return output_Feature_Merge_78123


if __name__ == "__main__":
    path_image = ""
    result = inference(path_image)
