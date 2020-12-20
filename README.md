# kitti-object-eval-python
**Note**: This is modified from [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)

Fast kitti object detection eval in python(finish eval in less than 10 second), support 2d/bev/3d/aos. , support coco-style AP. If you use command line interface, numba need some time to compile jit functions.

Add evaluation by distance.


## Results Look Like This

``` shell 
---------------  RESULTS by DISTANCE  ---------------

Car_3d_0-10m : 32.59071972040789
Car_3d_10-20m : 46.46858426610183
Car_3d_20-30m : 21.3083528208214
Car_3d_30-40m : 6.791346451532922
Car_3d_40-50m : 2.7378987255485088
Car_3d_50-60m : 0.02767400027674
Car_3d_60-70m : 0.0
Car_3d_70-80m : 0.0
Car_3d_>80m : 0.0
Car_bev_0-10m : 39.6144978136414
Car_bev_10-20m : 52.5811266753242
Car_bev_20-30m : 35.01909258485354
Car_bev_30-40m : 13.647555556083605
Car_bev_40-50m : 6.285789135650144
Car_bev_50-60m : 0.04013646397752358
Car_bev_60-70m : 0.0
Car_bev_70-80m : 0.0
Car_bev_>80m : 0.0
Car_image_0-10m : 92.10158703585401
Car_image_10-20m : 88.06226311139632
Car_image_20-30m : 83.52822664350285
Car_image_30-40m : 80.17999062869012
Car_image_40-50m : 69.97218274113382
Car_image_50-60m : 19.0304768941367
Car_image_60-70m : 0.0
Car_image_70-80m : 0.0
Car_image_>80m : 0.0
Car_3d_0-10m_R40 : 29.31513340992
Car_3d_10-20m_R40 : 44.09252932353496
Car_3d_20-30m_R40 : 19.955532715221036
Car_3d_30-40m_R40 : 5.757203040230914
Car_3d_40-50m_R40 : 2.353823931901796
Car_3d_50-60m_R40 : 0.008691314296187814
Car_3d_60-70m_R40 : 0.0
Car_3d_70-80m_R40 : 0.0
Car_bev_0-10m_R40 : 37.45320600519463
Car_bev_10-20m_R40 : 52.11212833874039
Car_bev_20-30m_R40 : 30.469522023182243
Car_bev_30-40m_R40 : 12.57455053812801
Car_bev_40-50m_R40 : 5.6498098255110465
Car_bev_50-60m_R40 : 0.012260021481349546
Car_bev_60-70m_R40 : 0.0
Car_bev_70-80m_R40 : 0.0
Car_image_0-10m_R40 : 95.128260775718
Car_image_10-20m_R40 : 89.27730261957814
Car_image_20-30m_R40 : 84.55434819624233
Car_image_30-40m_R40 : 81.18702126458646
Car_image_40-50m_R40 : 71.67394558575582
Car_image_50-60m_R40 : 17.31700642092629
Car_image_60-70m_R40 : 0.0
Car_image_70-80m_R40 : 0.0
```

## Dependencies
Only support python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in anaconda. Otherwise, please reference to this [page](https://github.com/numba/numba#custom-python-environments) to set up llvm and cuda for numba.
* Install by conda:
```
conda install -c numba cudatoolkit=x.x  (8.0, 9.0, 9.1, depend on your environment) 
```
## Usage
* commandline interface:
```
python evaluate.py evaluate --label_path=test/label2 --result_path=test/data_zh_079 --label_split_file=test/val.txt --current_class=0 --coco=False
```
* python interface:
```Python
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/path/to/your_result_folder"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/path/to/your_gt_label_folder"
gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer
```
