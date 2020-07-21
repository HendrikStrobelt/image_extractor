# Image extractor for PDFs using PubLayNet


### Setup Environment

- conda
- install image magick
- `cd weights;source download.sh;cd ..`

### Extract Image from PDF
After extraction you should have the following files available for 
a pdf `example_file.pdf`:

```
example_file.png  - the best representative image w.r.t. heuristic
(example_file_best_<number>.png) - [only if NOT shortcut] the images sorted by their representative value
```

#### From a set of PDFs:
```commandline
python extract_image.py --pdf proceedings_*.pdf --cleanup
```

Applies the image extraction to the PDF files, 
applies the simple histogram heuristic and cleans 
up all temporary files. Everything runs on a CPU machine.

#### From a set of PDFs - Just faster:
Runs the script on a CUDA GPU if available 
```commandline
python extract_image.py --device cuda --pdf proceedings_*.pdf --cleanup
```

#### From a set of PDFs - Even faster:
Uses the shortcut that if the first page contains an image,
it uses this one.  
```commandline
python extract_image.py --shortcut --device cuda --pdf proceedings_*.pdf --cleanup
```

### Extract Images from Paper images
No cleanup, no heuristic.. just plain image extraction:
```commandline
python extract_image.py --input pages_*.png 
```
  

### Usage

```commandline
usage: extract_image.py [-h] [--input INPUT [INPUT ...]]
                                 [--pdf PDF [PDF ...]]
                                 [--confidence-threshold CONFIDENCE_THRESHOLD]
                                 [--opts ...] [--config-file FILE]
                                 [--device DEVICE] [--weights WEIGHTS]
                                 [--overwrite] [--shortcut] [--cleanup]
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
  --shortcut            if image on first page use this (default: False)
  --cleanup             remove tmp files (default: False)
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