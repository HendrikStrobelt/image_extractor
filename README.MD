# Image extractor for PDFs using PubLayNet


### Setup Environment

1)  Create a conda env
```commandline
conda create -n image_ex python=3.7
conda activate image_ex
```
2) install [Pytorch and TorchVision](https://pytorch.org/get-started/locally/#start-locally). 
E.g.:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# or: pip install torch torchvision
```

3) install opencv and imagemagick
```
conda install -c conda-forge opencv

conda install -c conda-forge imagemagick
# or: apt install imagemagick
```

4) install [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#build-detectron2-from-source)

- [Prebuilds](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#install-pre-built-detectron2-linux-only) for Linux are easiest.
- For MacOS use:
```commandline
CC=clang CXX=clang++ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

5) Download weights: 

`cd weights;source download.sh;cd ..`



### Extract Image from PDF
After extraction you should have the following files available for 
a pdf `example_file.pdf`:

```
example_file.png  - the best representative image w.r.t. heuristic
(example_file_best_<number>.png) - [only if NOT shortcut] the images sorted by their representative value
```

#### From a set of PDFs:
```commandline
python extract_image.py --pdf proceedings_*.pdf --cleanup all
```

Applies the image extraction to the PDF files, 
applies the histogram heuristic plus first-page priority. 
It cleans up all temporary files. It detects automatically 
if running on cuda or cpu.


#### From a set of PDFs - Even faster:
Uses a shortcut that if the first page contains an image,
it uses this one and does not parse the rest.  
```commandline
python extract_image.py --firstpage shortcut --device cuda --pdf proceedings_*.pdf --cleanup
```

### Extract Images from Paper images
No cleanup, no heuristic.. just plain image extraction:
```commandline
python extract_image.py --input pages_*.png 
```
  

### Usage

```commandline
usage: extract_image.py [-h] [--input INPUT [INPUT ...]] [--pdf PDF [PDF ...]]
                        [--confidence-threshold CONFIDENCE_THRESHOLD]
                        [--opts ...] [--config-file FILE] [--device DEVICE]
                        [--weights WEIGHTS] [--overwrite]
                        [--firstpage FIRSTPAGE] [--cleanup CLEANUP]
                        [--accept N [N ...]]

Extract interesting images from PDFs or paper page images - using PubLayNet
and Detectron2 (pytorch) and a simple histogram heuristic.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...]
                        A list of space separated input images; or a single
                        glob pattern such as 'directory/*.jpg' (default: None)
  --pdf PDF [PDF ...]   A list of space separated PDF files.or a single glob
                        pattern such as 'directory/*.pdf' (default: None)
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum score for instance predictions to be shown
                        (default: 0.5)
  --opts ...            Modify config options using the command-line 'KEY
                        VALUE' pairs (default: [])
  --config-file FILE    path to config file (default:
                        configs/DLA_mask_rcnn_R_50_FPN_3x.yaml)
  --device DEVICE       run on device (cuda/cpu) (default: cpu)
  --weights WEIGHTS     run on device (default:
                        weights/DLA_mask_rcnn_R_50_FPN_3x_trimmed.pth)
  --overwrite           run on device (default: False)
  --firstpage FIRSTPAGE
                        if image on first page give it high priority --
                        values: ['shortcut','prio','none'] (default: prio)
  --cleanup CLEANUP     remove tmp files [images, pages, all] (default: none)
  --accept N [N ...]    accepted classes (default: [4])
```

### License

Apache 2.0

### Acknowledgements

This code is heavily based on these repositories:
- [https://github.com/ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
- [https://github.com/hpanwar08/detectron2](https://github.com/hpanwar08/detectron2)
- [https://github.com/Mini-Conf/image-extraction](https://github.com/Mini-Conf/image-extraction)

Thank you dear contributors !! Please contact me if you like a different mention
