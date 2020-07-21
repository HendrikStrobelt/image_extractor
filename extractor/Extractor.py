import glob
import json
import os
import time

import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import tqdm

# inspired by VisualizationDemo
from fvcore.common.file_io import PathManager


ImageFile.LOAD_TRUNCATED_IMAGES = True

class Extractor(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):

        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
        """

        return self.predictor(image)

    # TODO: remove args
    def run_extraction_for_images(self, inputs, accept, args, logger):
        all_cropped_imgs = []
        for path in tqdm.tqdm(inputs):  # , disable=not args.output):
            self.process_single_image(path, all_cropped_imgs, args, accept,
                                      logger)
        return all_cropped_imgs

    # TODO: remove args
    def process_single_image(self, path, all_cropped_imgs, args, accept, logger):
        prefix = os.path.splitext(path)[0]
        result_file_name = '{}_scanned.json'.format(prefix)
        if os.path.exists(result_file_name) and not args.overwrite:
            #  just add cached segments
            for cropped_img_name in glob.glob(prefix + '_????_type?.png'):
                logger.info("loading {}".format(cropped_img_name))
                all_cropped_imgs.append(Image.open(cropped_img_name))
        else:
            # use PIL, to be consistent with evaluation
            np_img, pil_img = self.load_image(path, format="BGR")
            start_time = time.time()
            predictions = self.run_on_image(np_img)
            # logger.info(predictions["instances"].get_fields().keys())
            # logger.info(predictions["instances"].get("pred_classes"))
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(
                        len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            info_dict, crop_imgs = self.crop_regions_and_parse(
                predictions['instances'],
                pil_img, prefix, accept)
            all_cropped_imgs.extend(crop_imgs)
            with open(result_file_name, 'w') as f:
                json.dump(info_dict, f)

    def crop_regions_and_parse(self, predictions, img: Image.Image, prefix, accepted):
        boxes = predictions.pred_boxes if predictions.has(
            "pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has(
            "pred_classes") else None
        num_predictions = len(predictions)

        collected_imgs = []
        max_x, max_y = img.size
        box_list = None
        if boxes:
            box_list = boxes.tensor.tolist()
            for i in range(num_predictions):
                my_class = classes[i].tolist()
                if (accepted == []) or (my_class in accepted):
                    # print(box_list[i], classes[i].tolist())
                    x0, y0, x1, y1 = box_list[i]
                    x0, y0, x1, y1 = max([x0 - 20, 0]), \
                                     max([y0 - 20]), \
                                     min([x1 + 20, max_x]), \
                                     min([y1 + 20, max_y])

                    crop_img = img.crop([x0, y0, x1, y1])
                    collected_imgs.append(crop_img)
                    crop_img.save(
                        '{}_{:04d}_type{}.png'.format(prefix, i, my_class))

        return {
                   'boxes': box_list,
                   'classes': classes.tolist(),
                   'scores': scores.tolist()
               }, collected_imgs

    def load_image(self, file_name, format):
        with PathManager.open(file_name, "rb") as f:
            image: Image.Image = Image.open(f)
            # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            # noinspection PyTypeChecker
            image = _apply_exif_orientation(image)
            # noinspection PyTypeChecker
            return convert_PIL_to_numpy(image, format), image
