


# Real deep learning can generalise to more than one species: A Comparative Three Species Whole Slide Image Dataset

This repository contains code to replicate the results from the paper:
[Real deep learning can generalise to more than one species: A Comparative Three Species Whole Slide Image Dataset](https://arxiv.org/abs/2108.08529) and links to corresponding jupyter notebooks. 
The dataset can be examined at [EXACT](https://exact.cs.fau.de/) with the username ```SDATA_EIPH_2021``` and the password ```SDATA_ALBA```

## Start and Structure

- Please install the [requirements.txt](requirements.txt) ```pip install -r requirements.txt```
- Download the slides [Download.ipynb](Download.ipynb)
- Install Openslide for Linux ```!apt-get install python3-openslide``` or Windows https://openslide.org/download/
- The folder [Statistics](Statistics) contains notebooks which analyse the dataset annotations
and general information about the slides
- [Inference](Inference) contains code to train the described models and perform inference on
slides.
- [Regression](Regression) trains the regression models to predict a continuous EIPH grade and is used for creating the density
maps
- [Cluster](Cluster) contains code to create custom annotation maps and synchronise the generated images and annotations with
EXACT.

# Troubleshooting

If you are facing the following error message at GitHub 

```
Sorry, something went wrong. Reload?
```

Please use:

https://nbviewer.jupyter.org/github/ChristianMarzahl/EIPH_WSI/tree/master/SDATA/


## Object detection library
For object detection on whole slide images (WSI), we use code from this  https://github.com/ChristianMarzahl/ObjectDetection repository.
If you are using the repository or parts thereof, please cite the corresponding [paper](https://www.nature.com/articles/s41598-020-65958-2):
```
@article{marzahl2020deep,
  title={Deep learning-based quantification of pulmonary hemosiderophages in cytology slides},
  author={Marzahl, Christian and Aubreville, Marc and Bertram, Christof A and Stayt, Jason and Jasensky, Anne-Katherine and Bartenschlager, Florian and Fragoso-Garcia, Marco and Barton, Ann K and Elsemann, Svenja and Jabari, Samir and Jens, Krauth and Prathmesh, Madhu and Jörn, Voigt and Jenny, Hill and Robert, Klopfleisch and Andreas, Maier },
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={1--10},
  year={2020},
  publisher={Nature Publishing Group}
}
```

## Abstract

Pulmonary hemorrhage (P-Hem) can have various causes including drug abuse, physical exercise, premature birth, leukaemia, autoimmune disorders and immunodeficiencies among multiple species. Cytology of bronchoalveolar lavage fluid using a scoring system which classifies hermosiderophages into five grades is considered the most sensitive diagnostic method independent of the species.
We introduce a novel fully annotated multi-species P-Hem data-set of 74 cytology whole slide images (WSI) with human, feline  and equine samples. To create this high quality and high quantity data-set, we developed an annotation pipeline combining human expertise with deep learning and data visualisation techniques. Starting with a deep learning-based object detection approach trained on 17 equine WSI annotated by experts and afterwards applied to the remaining 47 equine, human and feline WSIs.
The resulting annotations are efficiently semi-automatically screened for errors on multiple types of specialised annotation maps and finally validated by trained pathologists. Our data-set contains 297,383 hemosiderophages in the grades from zero to four and is one of the largest publicly available WSI data-set in respect to number of annotations, scanned area and number of species covered for one pathology. We validated our data-set with a species-wise three by three cross validation resulting in mean Average Precisions (mAP) ranging from 0.77 mAP for training on feline and validating on equine slides up to 0.88 for equine versus human. And a mean intra species mAP of 0.90 (min = 0.88, max = 0.91) showing that equine can be used to substitute human samples. 


## Overview

![OVerview](Paper/Overview.svg)
Overview of the macrophage annotation and validation pipeline: The RetinaNet object-detection model trained on
16 equine slides4
is used to perform inference on the remaining slides, followed by a semi-automatic clustering step which
clusters cells by size. Error-prone cells are highlighted and can then be efficiently deleted by a human expert. Afterwards, a
human expert screens all WSI to increase the dataset consistency. Finally, a regression-based clustering system is applied to
support experts searching for misclassifications of the hemosiderin grade.


## Species confusion matrix

![OVerview](Statistics/CrossValidation/SpeciesConfusionMatrix.svg)
Results of the ablation study using our customised RetinaNet object detector on an increasing number of humane,
equine and feline training patches of size 1024 x 1024 pixel from one WSI or up to five complete WSIs. The boxes represent
the total number of hemosiderophages used for training in combination with the mAP graphs for each species.

## Species confusion matrix

![OVerview](Statistics/AblationStudy/AblationStudy_log.svg)
Each of the nine figures show on the left the species source training domain and on the top the species target domain
with the obtained mAP. Green bounding boxes represent grade zero hemosiderophages while red show grade one