import time
import fire

import kitti_common as kitti
from eval import get_official_eval_result_by_distance, get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
    
def printer_decorator(func):
    def print_results(result):
        for r in result:
            if type(r)==str:
                print(r)
            elif type(r)==list:
                print_results(r)
            elif type(r)==dict:
                for kw in r:
                    print(kw,':',r[kw])

    def printer(*args, **kwargs):
        default_res, distance_res = func(*args, **kwargs)
        print("--------------- RESULTS by DIFFICULTY ---------------")
        print_results(default_res)
        print("---------------  RESULTS by DISTANCE  ---------------")
        print_results(distance_res)

    return printer

@printer_decorator
def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return [get_official_eval_result(gt_annos, dt_annos, current_class), get_official_eval_result_by_distance(gt_annos, dt_annos, current_class)]



        

if __name__ == '__main__':
    fire.Fire()
