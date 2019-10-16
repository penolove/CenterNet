import _init_paths

import arrow

import numpy as np
from bistiming import Stopwatch
import argparse

from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.config import BoundedBoxObject
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image

from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset
from lib.utils.debugger import pascal_class_name, coco_class_name

CLASSES_MAPPING = {"pascal": pascal_class_name, "coco": coco_class_name}


class ObjectDetectorOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset", default="coco", help="coco | pascal")
        self.parser.add_argument(
            "--load_model", default="", help="path to pretrained model"
        )

        self.parser.add_argument(
            "--gpus", default="0", help="-1 for CPU, use comma for multiple gpus"
        )
        self.parser.add_argument(
            "--arch",
            default="dla_34",
            help="model architecture. Currently tested"
            "res_18 | res_101 | resdcn_18 | resdcn_101 |"
            "dlav0_34 | dla_34 | hourglass",
        )

        self.parser.add_argument(
            "--vis_thresh", type=float, default=0.3, help="visualization threshold."
        )

        self.parser.add_argument(
            "--down_ratio",
            type=int,
            default=4,
            help="output stride. Currently only supports 4.",
        )
        self.parser.add_argument(
            "--input_res",
            type=int,
            default=-1,
            help="input height and width. -1 for default from "
            "dataset. Will be overriden by input_h | input_w",
        )
        self.parser.add_argument(
            "--input_h",
            type=int,
            default=-1,
            help="input height. -1 for default from dataset.",
        )
        self.parser.add_argument(
            "--input_w",
            type=int,
            default=-1,
            help="input width. -1 for default from dataset.",
        )
        self.parser.add_argument(
            "--cat_spec_wh",
            action="store_true",
            help="category specific bounding box size.",
        )
        self.parser.add_argument(
            "--not_reg_offset", action="store_true", help="not regress local offset."
        )

        # test
        self.parser.add_argument(
            "--flip_test", action="store_true", help="flip data augmentation."
        )
        self.parser.add_argument(
            "--test_scales",
            type=str,
            default="1",
            help="multi scale test augmentation.",
        )
        self.parser.add_argument(
            "--nms", action="store_true", help="run nms in testing."
        )
        self.parser.add_argument(
            "--K", type=int, default=100, help="max number of output objects."
        )
        self.parser.add_argument(
            "--not_prefetch_test",
            action="store_true",
            help="not use parallal data pre-processing.",
        )
        self.parser.add_argument(
            "--fix_res",
            action="store_true",
            help="fix testing resolution or keep " "the original resolution",
        )
        self.parser.add_argument(
            "--keep_res",
            action="store_true",
            help="keep the original resolution" " during validation.",
        )

        self.parser.add_argument(
            "--debug",
            type=int,
            default=0,
            help="level of visualization."
            "1: only show the final detection results"
            "2: show the network output features"
            "3: use matplot to display"  # useful when lunching training with ipython notebook
            "4: save all visualizations to disk",
        )

        self.parser.add_argument(
            "--debugger_theme", default="white", choices=["white", "black"]
        )

    @property
    def task(self):
        return "ctdet"

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(",")]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        opt.head_conv = 256
        opt.pad = 31
        opt.num_stacks = 1

        dataset = get_dataset(opt.dataset, self.task)
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        opt.test_scales = [float(i) for i in opt.test_scales.split(",")]
        opt.fix_res = True

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)
        opt.reg_offset = not opt.not_reg_offset

        opt.heads = {
            "hm": opt.num_classes,
            "wh": 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
        }
        if not opt.not_reg_offset:
            opt.heads.update({"reg": 2})

        return opt


class CenterNetDetectorWrapper(ObjectDetector):
    def __init__(self, opt):
        Detector = detector_factory["ctdet"]
        self.detector = Detector(opt)
        self.classes = CLASSES_MAPPING[opt.dataset]

    def detect(self, image_obj) -> DetectionResult:
        image_raw_width = image_obj.pil_image_obj.width
        image_raw_height = image_obj.pil_image_obj.height
        detected_objects = []
        result = self.detector.run(np.array(image_obj.pil_image_obj))
        for label_class in range(1, self.detector.num_classes + 1):
            for out in result["results"][label_class]:
                bbox = out[:4]
                score = out[4]
                if score > self.detector.opt.vis_thresh:
                    label = self.classes[label_class - 1]
                    x_coord, y_coord, width, height = bbox
                    x1 = max(0, np.floor(x_coord + 0.5).astype(int))
                    y1 = max(0, np.floor(y_coord + 0.5).astype(int))
                    x2 = min(
                        image_raw_width, np.floor(x_coord + width + 0.5).astype(int)
                    )
                    y2 = min(
                        image_raw_height, np.floor(y_coord + height + 0.5).astype(int)
                    )

                    # handle the edge case of padding space
                    x1 = min(image_raw_width, x1)
                    x2 = min(image_raw_width, x2)
                    if x1 == x2:
                        continue
                    y1 = min(image_raw_height, y1)
                    y2 = min(image_raw_height, y2)
                    if y1 == y2:
                        continue
                    detected_objects.append(
                        BoundedBoxObject(x1, y1, x2, y2, label, score, "")
                    )

        image_dict = {
            "image_id": image_obj.image_id,
            "detected_objects": detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    @property
    def valid_labels(self):
        return set(self.classes)


if __name__ == "__main__":
    opt = ObjectDetectorOpts().parse()
    object_detector = CenterNetDetectorWrapper(opt)
    raw_image_path = "demo/test_image.jpg"
    image_id = ImageId(
        channel="demo", timestamp=arrow.now().timestamp, file_format="jpg"
    )
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with Stopwatch("Running inference on image {}...".format(raw_image_path)):
        detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")

