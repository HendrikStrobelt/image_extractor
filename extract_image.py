import argparse
import glob
import json
import os

from math import log

import torch
from PIL import ImageFile
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import numpy as np

from extractor.PdfExtractor import PdfExtractor
from extractor.Extractor import Extractor

__author__ = "Hendrik Strobelt, Alexander M. Rush"

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        default='cuda' if torch.cuda.is_available() else 'cpu'
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
        "--firstpage",
        help="if image on first page give it high priority "
             "-- values: ['shortcut','prio','none']",
        default='prio'

    )
    parser.add_argument(
        "--cleanup",
        help="remove tmp files [images, pages, all]",
        default='none'
    )
    parser.add_argument(
        '--accept',
        metavar='N',
        type=int,
        nargs='+',
        default=[4],
        help='accepted classes')

    return parser

    # labels = _create_text_labels(classes, scores,
    #                              self.metadata.get("thing_classes", None))
    # keypoints = predictions.pred_keypoints if predictions.has(
    #     "pred_keypoints") else None


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
        extractor.run_extraction_for_images(args.input, args.accept, args, logger)
    elif args.pdf:
        if len(args.pdf) == 1:
            args.pdf = glob.glob(os.path.expanduser(args.pdf[0]))
            assert args.pdf, "The input path(s) was not found"

        for pdf_path in args.pdf:
            logger.info(" =====> processing {}".format(pdf_path))
            logger.info('Rendering pdf pages....')
            prefix = os.path.splitext(pdf_path)[0]
            master_file_name = prefix + '.json'

            if (not args.overwrite and os.path.exists(master_file_name)) \
                    or PdfExtractor.pdf_to_imgs(pdf_path) == 0:
                logger.info('.. done. Finding images...')

                page_files = glob.glob(os.path.expanduser(os.path.splitext(pdf_path)[0] + '-????.png'))
                page_files.sort()
                with open(master_file_name, 'w') as f:
                    json.dump({"pages": page_files}, f)

                # split to first page and rest
                page0, *rest_page_files = page_files

                cropped_images = []

                extractor.process_single_image(page0, cropped_images, args, accept=args.accept, logger=logger)

                # == short cut if image on first page ==
                best_candidate_found = False
                if len(cropped_images) > 0 and (not (args.firstpage == 'none')):
                    cropped_images[0].save(prefix + '.png')
                    best_candidate_found = True
                    logger.info("chose first page")

                if args.firstpage == 'shortcut':
                    logger.info('.. shortcut for {}. Done.'.format(pdf_path))
                else:
                    cropped_images_add = extractor.run_extraction_for_images(rest_page_files, args.accept, args, logger)
                    cropped_images.extend(cropped_images_add)
                    disp_values = list(map(lambda x: get_histogram_dispersion(x.histogram()), cropped_images))
                    sorted_indices = np.argsort(np.array(disp_values))[::-1].tolist()
                    for i, index in enumerate(sorted_indices):
                        if i == 0 and not best_candidate_found:
                            cropped_images[index].save(prefix + '.png')
                        cropped_images[index].save('{}_best_{:04d}.png'.format(prefix, i))
                    logger.info('.. done.')

                # === CLEANUP ===
                if not args.cleanup == "none":
                    if args.cleanup == "pages" or args.cleanup == 'all':
                        for del_file in page_files:
                            os.remove(del_file)
                    if args.cleanup == "images" or args.cleanup == 'all':
                        for del_file in glob.glob(
                                '{}-????_scanned.json'.format(prefix)):
                            os.remove(del_file)
                        for del_file in glob.glob(
                                '{}-????_????_type?.png'.format(prefix)):
                            os.remove(del_file)

            else:
                logger.error('Something went wrong.')
