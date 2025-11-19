from pathlib import Path

import SimpleITK

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


import glob
import SimpleITK
from pathlib import Path
import os
from skimage.measure import regionprops, regionprops_table
import numpy as np
import subprocess
import sys
import shutil

import random
import torch
import numpy as np

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings('ignore')


def make_dir(dst):
    try:
        os.mkdir(dst)
    except:
        return

def masks_to_boxes(to_be_seg_img, lung_img):
    #Careful, there can be multiple segments, then we need multiple bboxes
    binary_filt = SimpleITK.BinaryThresholdImageFilter()
    binary_filt.SetOutsideValue(0)
    binary_filt.SetUpperThreshold(100)
    binary_filt.SetLowerThreshold(1)
    lung_img = binary_filt.Execute(lung_img)
    lung_img = lung_img[1:-1,1:-1,1:-1]

    labelimfilter = SimpleITK.LabelShapeStatisticsImageFilter()
    labelimfilter.Execute(lung_img)
    lung_bbox = labelimfilter.GetBoundingBox(1)
    region_filter = SimpleITK.RegionOfInterestImageFilter()
    region_filter.SetRegionOfInterest(lung_bbox)
    crop_img = region_filter.Execute(to_be_seg_img)
    return crop_img


def resample_seg(gtv_stk, ref_stk):
    gtv_padded_stk = SimpleITK.Resample(gtv_stk, ref_stk.GetSize(),
                                          SimpleITK.Transform(),
                                          SimpleITK.sitkNearestNeighbor,
                                          ref_stk.GetOrigin(),
                                          ref_stk.GetSpacing(),
                                          ref_stk.GetDirection(),
                                          0,
                                          ref_stk.GetPixelID())
    return gtv_padded_stk

def setup_nnunet_v2():
    subprocess.check_call([sys.executable, '-m', 'pip', 'show', 'nnunetv2'])  # dependency conflict of nnUntv2 and totalsegmentator
    code_change_dir = Path('/opt/algorithm/nnunet/nnunet_code_changes')
    dst_code_files = Path('/home/user/.local/lib/python3.10/site-packages/')
    shutil.copytree(code_change_dir / 'nnunetv2', dst_code_files / 'nnunetv2', dirs_exist_ok=True)


class Lnq2023(SegmentationAlgorithm):
    def __init__(self):
        output_path = Path('/output/images/mediastinal-lymph-node-segmentation/')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            input_path=Path('/input/images/mediastinal-ct/'),
            output_path=output_path,
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        os.environ["nnUNet_raw"] = '/opt/algorithm/nnunet/nnUNet_raw/'
        os.environ["nnUNet_preprocessed"] = '/opt/algorithm/nnunet/nnUNet_preprocessed/'
        os.environ["nnUNet_results"] = '/opt/algorithm/nnunet/nnUNet_results/'
        make_dir(os.environ["nnUNet_raw"])
        make_dir(os.environ["nnUNet_preprocessed"])
        make_dir(os.environ["nnUNet_results"])
        self.work_dir = Path('/opt/app/tmp')

        # journal model 7 trained with default nnUNet params and no postprocessing
        self.model_ckp = Path(os.environ["nnUNet_results"]) / 'Dataset050_LNQ_Bouget_NSCLC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth'

        print('Using Model checkpoint:')
        print(self.model_ckp)
        os.mkdir(self.work_dir)

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        import time
        start = time.time()

        input_image_path = self.work_dir / 'full_ct_img.nii.gz'
        SimpleITK.WriteImage(input_image, input_image_path)
        print('IMG SIZE')
        print(input_image.GetSize())

        print('IMG SIZE NP')
        print(SimpleITK.GetArrayFromImage(input_image).shape)

        img_id = 'full_ct_img'

        # apply lung cropping
        total_path = Path(self.work_dir) / 'totalsegmentator_structures'
        os.mkdir(total_path)

        cmd_total = 'TotalSegmentator -i ' + str(input_image_path) + ' -o  ' + str(
            total_path) + ' --roi_subset lung_lower_lobe_left lung_upper_lobe_left lung_lower_lobe_right lung_upper_lobe_right lung_middle_lobe_right'
        os.system("conda run -n totalsegmentator " + cmd_total)

        print(os.listdir(total_path))

        cmd_total = 'totalseg_combine_masks -i  ' + str(total_path) + ' -o ' + str(total_path) + '/lung.nii.gz -m  lung'
        os.system("conda run -n totalsegmentator " + cmd_total)

        # read input ct and lung region
        inference_ct_stk = SimpleITK.ReadImage(input_image_path)
        lung_stk = SimpleITK.ReadImage(total_path / 'lung.nii.gz', SimpleITK.sitkInt8)

        # crop ct input to lung region
        inference_ct_cropped_stk = masks_to_boxes(inference_ct_stk, lung_stk)
        save_path = total_path / 'lung_CT'
        os.mkdir(save_path)
        SimpleITK.WriteImage(inference_ct_cropped_stk, str(save_path / str(img_id + '_0000.nii.gz')))
        print(os.listdir(save_path))

        # Adjust python env and system variables
        setup_nnunet_v2()
        inference_path = total_path / 'inference'
        os.mkdir(inference_path)

        # apply nnunet on cropped ct
        nnunet_cmd = 'nnUNetv2_predict -i ' + str(save_path) + ' -o ' + str(
            inference_path) + ' -d 50 -c 3d_fullres -f all -chk ' + str(self.model_ckp)
        os.system("conda run -n nnunet_raw " + nnunet_cmd)
        print(os.listdir(inference_path))
        print("os.listdir(os.environ[nnUNet_preprocessed])")
        print(os.listdir(os.environ["nnUNet_preprocessed"] + "Dataset029_all_at_once"))

        seg_file = glob.glob(str(inference_path / str(img_id + '*.*')))[0]
        labelmap_stk = SimpleITK.ReadImage(seg_file, SimpleITK.sitkInt8)


        # save filtered lymph node segmentations
        ref_stk = SimpleITK.ReadImage(input_image_path)
        labelmap_stk = resample_seg(labelmap_stk, ref_stk)

        end = time.time()
        print('Time needed for execution in seconds:')
        print(end - start)

        return labelmap_stk


if __name__ == "__main__":
    Lnq2023().process()

