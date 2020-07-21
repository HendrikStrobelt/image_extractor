import argparse
import glob
import json
import os
import time
from math import log
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, \
    convert_PIL_to_numpy
from detectron2.utils.logger import setup_logger
import tqdm
import numpy as np
from fvcore.common.file_io import PathManager
from extractor.Extractor import Extractor

__author__ = "Hendrik Strobelt, Alexander M. Rush"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    return cfg


def get_parser():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract interesting images from PDFs or paper page images "
                    "- using PubLayNet and Detectron2 (pytorch) and a simple "
                    "histogram heuristic. ")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        help="A list of space separated PDF files."
             "or a single glob pattern such as 'directory/*.pdf'",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config-file",
        default="configs/DLA_mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--device",
        help="run on device (cuda/cpu)",
        default='cpu'
    )
    parser.add_argument(
        "--weights",
        help="run on device",
        default='weights/DLA_mask_rcnn_R_50_FPN_3x_trimmed.pth'
    )
    parser.add_argument(
        "--overwrite",
        help="run on device",
        action='store_true'
    )
    parser.add_argument(
        "--shortcut",
        help="if image on first page use this",
        action='store_true'
    )
    parser.add_argument(
        "--cleanup",
        help="remove tmp files",
        action='store_true'
    )
    parser.add_argument(
        '--accept',
        metavar='N',
        type=int,
        nargs='+',
        default=[4],
        help='accepted classes')

    return parser


def crop_regions_and_parse(predictions, img: Image.Image, prefix, accepted):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
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
    # labels = _create_text_labels(classes, scores,
    #                              self.metadata.get("thing_classes", None))
    # keypoints = predictions.pred_keypoints if predictions.has(
    #     "pred_keypoints") else None


def load_image(file_name, format):
    with PathManager.open(file_name, "rb") as f:
        image: Image.Image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        # noinspection PyTypeChecker
        image = _apply_exif_orientation(image)
        # noinspection PyTypeChecker
        return convert_PIL_to_numpy(image, format), image


def run_extraction_for_images(inputs, accept, args):
    all_cropped_imgs = []
    for path in tqdm.tqdm(inputs):  # , disable=not args.output):
        process_single_image(path, all_cropped_imgs, args, accept)
    return all_cropped_imgs


def process_single_image(path, all_cropped_imgs, args, accept):
    prefix = os.path.splitext(path)[0]
    result_file_name = '{}_scanned.json'.format(prefix)
    if os.path.exists(result_file_name) and not args.overwrite:
        #  just add cached segments
        for cropped_img_name in glob.glob(prefix + '_????_type?.png'):
            logger.info("loading {}".format(cropped_img_name))
            all_cropped_imgs.append(Image.open(cropped_img_name))
    else:
        # use PIL, to be consistent with evaluation
        np_img, pil_img = load_image(path, format="BGR")
        start_time = time.time()
        predictions = extractor.run_on_image(np_img)
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

        info_dict, crop_imgs = crop_regions_and_parse(
            predictions['instances'],
            pil_img, prefix, accept)
        all_cropped_imgs.extend(crop_imgs)
        with open(result_file_name, 'w') as f:
            json.dump(info_dict, f)


def pdf_to_imgs(pdf_file_name):
    command = (
            "convert -background white  -alpha remove -alpha off -density 200 '"
            + pdf_file_name
            + "'[0-12]  png24:"
            + os.path.splitext(pdf_file_name)[0]
            + "-%04d.png"
    )
    return os.system(command)


def get_histogram_dispersion(histogram):
    log2 = lambda x: log(x) / log(2)

    total = len(histogram)
    counts = {}
    for item in histogram:
        counts.setdefault(item, 0)
        counts[item] += 1

    ent = 0
    for i in counts:
        p = float(counts[i]) / total
        ent -= p * log2(p)
    return -ent * log2(1 / ent)


if __name__ == '__main__':
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    extractor = Extractor(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        run_extraction_for_images(args.input, args.accept, args)
    elif args.pdf:
        if len(args.pdf) == 1:
            args.pdf = glob.glob(os.path.expanduser(args.pdf[0]))
            assert args.pdf, "The input path(s) was not found"

        for pdf_path in args.pdf:
            logger.info(" =====> processing {}".format(pdf_path))
            logger.info('Rendering pdf pages....')
            if pdf_to_imgs(pdf_path) == 0:
                logger.info('.. done. Finding images...')
                page_files = glob.glob(os.path.expanduser(
                    os.path.splitext(pdf_path)[0] + '-????.png'))
                page_files.sort()

                prefix = os.path.splitext(pdf_path)[0]

                page0, *rest_page_files = page_files

                cropped_images = []
                process_single_image(page0, cropped_images, args,
                                     accept=args.accept)

                # == short cut if image on first page ==
                if len(cropped_images) > 0 and args.shortcut:
                    cropped_images[0].save(prefix + '.png')
                    logger.info('.. shortcut for {}. Done.'.format(pdf_path))
                else:
                    cropped_images_add = run_extraction_for_images(
                        rest_page_files, args.accept, args)
                    cropped_images.extend(cropped_images_add)
                    disp_values = list(
                        map(lambda x: get_histogram_dispersion(x.histogram()),
                            cropped_images))
                    sorted_indices = np.argsort(np.array(disp_values))[
                                     ::-1].tolist()
                    for i, index in enumerate(sorted_indices):
                        if i == 0:
                            cropped_images[index].save(prefix + '.png')
                        cropped_images[index].save(
                            '{}_best_{:04d}.png'.format(prefix, i))
                    logger.info('.. done.')

                # === CLEANUP ===
                if args.cleanup:
                    for del_file in page_files:
                        os.remove(del_file)
                    for del_file in glob.glob(
                            '{}-????_scanned.json'.format(prefix)):
                        os.remove(del_file)
                    for del_file in glob.glob(
                            '{}-????_????_type?.png'.format(prefix)):
                        os.remove(del_file)

            else:
                logger.error('Something went wrong.')
