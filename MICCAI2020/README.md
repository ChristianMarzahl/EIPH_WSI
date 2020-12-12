# Data for MICCAI 2020 Submission

Deep-learning-based pipelines have recently shown the potential to revolutionalize microscopy image diagnostics by providing visual augmentations and evaluations to a trained pathology expert. However, to match human performance, the methods rely on the availability of vast amounts of high-quality labeled data, which poses a significant challenge today. To circumvent this, augmented labeling methods, also known as expert-algorithm-collaboration, have recently become popular. Yet, potential biases introduced by this operation mode and their effects for training deep neuronal networks are not entirely understood. 

This work aims to shed light on some of the effects by providing a case study for three relevant diagnostic settings: First, the labeling of different, well separable classes of cells on a cytology slide. Second, the grading of stained cells with regards to their approximated dye-concentration. Lastly, mitotic figure detection - a prominent task in histopathologic tumor grading. Ten trained pathology experts performed the tasks first  without and later with computer-generated augmentation. To investigate different biasing effects, we intentionally introduced errors to the augmentation. Furthermore, we developed a novel loss function which incorporates the experts' annotation consensus to train a deep learning classifier.


In total, ten pathology experts annotated 26,015 cells on 1,200 images in this novel annotation study. Backed by this extensive data set, we found that the concordance with multi-expert consensus was significantly increased in the computer-aided setting, versus the annotation from scratch. However, a significant percentage of the deliberately introduced false labels was not identified by the experts. Additionally, we showed that our loss function profited from multiple experts and outperformed conventional loss functions. At the same time, systematic errors did not lead to a deterioration of the trained classifier accuracy.

# Results

We employed the open-source platform [EXACT](https://github.com/ChristianMarzahl/Exact) to host our experiments. Additionaly, the study is staying online for further contributions. 

```
User: StudyJan2020
PW: Alba2020
```

The annotations are under [Results](Results/)


## EIPH:

[Notebook](JanuaryStudyEIPH.ipynb)

## Mitotic Figures:

[Notebook](JanuaryStudyMitosen.ipynb)

## Astma: 

[Notebook](JanuaryStudyMitosen.ipynb)

## Combined:

[Notebook](JanuaryStudyNoLabelBox.ipynb)

## Deep Learning for patch classification

### Results:
[Notebook](PatchClassification/LossAccuracy.ipynb)


### Training:

#### Annotation_Mode

[Notebooks:](PatchClassification/Annotation)

#### Expert-Algorithm-Mode

[Notebooks:](PatchClassification/ExpertAlgorithm)

## Anova:

[EXCEL](Anova.xlsx)

