# [Real deep learning can generalise to more than one species: A Comparative Three Species Whole Slide Image Dataset]

This repository contains code to replicate the results from the paper:
[Real deep learning can generalise to more than one species: A Comparative Three Species Whole Slide Image Dataset]() and links to corresponding jupyter notebooks. 

## Abstract

Pulmonary hemorrhage (P-Hem) can have various causes including drug abuse, physical exercise, premature birth, leukaemia, autoimmune disorders and immunodeficiencies among multiple species. Cytology of bronchoalveolar lavage fluid using a scoring system which classifies hermosiderophages into five grades is considered the most sensitive diagnostic method independent of the species.
We introduce a novel fully annotated multi-species P-Hem data-set of 74 cytology whole slide images (WSI) with human, feline  and equine samples. To create this high quality and high quantity data-set, we developed an annotation pipeline combining human expertise with deep learning and data visualisation techniques. Starting with a deep learning-based object detection approach trained on 17 equine WSI annotated by experts and afterwards applied to the remaining 47 equine, human and feline WSIs.
The resulting annotations are efficiently semi-automatically screened for errors on multiple types of specialised annotation maps and finally validated by trained pathologists. Our data-set contains 297,383 hemosiderophages in the grades from zero to four and is one of the largest publicly available WSI data-set in respect to number of annotations, scanned area and number of species covered for one pathology. We validated our data-set with a species-wise three by three cross validation resulting in mean Average Precisions (mAP) ranging from 0.77 mAP for training on feline and validating on equine slides up to 0.88 for equine versus human. And a mean intra species mAP of 0.90 (min = 0.88, max = 0.91) showing that equine can be used to substitute human samples. 


## Overview

![OVerview](Paper/Overview.svg)


## Species confusion matrix

![OVerview](Statistics/CrossValidation/SpeciesConfusionMatrix.svg)

## Species confusion matrix

![OVerview](Statistics/AblationStudy/AblationStudy_log.svg)