# Deep Learning-Based Quantification of PulmonaryHemosiderophages in Cytology Slides


This repository contains code to replicate the results from the paper:
[Deep Learning-Based Quantification of PulmonaryHemosiderophages in Cytology Slides](https://arxiv.org/abs/1908.04767) and links to corresponding jupyter notebooks. 


<a href="http://www.youtube.com/watch?feature=player_embedded&v=6azMAYpsyRw
" target="_blank"><img src="http://img.youtube.com/vi/6azMAYpsyRw/0.jpg" 
alt="Object Detection" width="240" height="180" border="10" /></a>


#### Regression:

The following [notebook](Regression/baseline.ipynb) generates the cell-based regression scores per cell and trains the model. 
![alt text](ReadmeImages/RegressionCellScores.png "Cell based regression results.")




The following [notebook](Regression/baseline.ipynb) generates the cell-based regression map. 
![alt text](ReadmeImages/Density2.png "Cell based regression results.")



#### QuadTree:

The following [notebook](QuadTree/QTree.ipynb) generates the quad tree. 
![alt text](ReadmeImages/ObjectDetectionWithQuadTree.png "Quad tree with object detection results.")


#### Object Detection

The following [notebook](Detection/baseline-level0.ipynb) traines RetinaNet on the EIPH dataset with the following results. 
![alt text](ReadmeImages/Cells1.png "Quad tree with object detection results.")


#### Results 

The following [notebook](Statistics/ClassificationResults.ipynb) calculates some metrics used in the paper. 
![alt text](ReadmeImages/ConfusionMatrix.png "Confusion Matrix")

