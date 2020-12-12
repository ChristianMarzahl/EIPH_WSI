import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import matplotlib
from math import pow, sqrt, ceil
from collections import OrderedDict
from sklearn.neighbors import KDTree

from Analysis.expert import ProjectType, DatasetType

class Statistics:
    
    def __init__(self, participants):

        self.participants = participants

    def calc_statistics(self, iou_thresh=0.25):

        data = []

        for participant in self.participants:

            ground_truths = [temp for temp in self.participants if temp.ProjectType == ProjectType.GroundTruth and temp.DatasetType == participant.DatasetType]
            if len(ground_truths) > 0:
                ground_truth = ground_truths[0]
            else:
                print(participant)

            order = -1
            if "_" in participant.expert:
                order = int(participant.expert.split("_")[1])

            miou = participant.calc_MIoU(ground_truth)
            data.append([participant.expert, participant.DatasetType, participant.ProjectType ,participant.total_annotations, participant.mean_seconds_to_label, miou, order])

        return pd.DataFrame(data, columns=['Name', 'Dataset', 'ProjectType', 'Nr. Annotations', 'Seconds', 'mIoU', 'Order'])


    def get_file_annotations(self, image_name):

        data = []

        for participant in self.participants:
            if image_name not in participant.Images:
                continue
            image = participant.Images[image_name]

            annotations = image.Annotations

            for anno in annotations:
                data.append([participant.expert, anno.Vector, anno.Label, participant.ProjectType, participant.DatasetType])

        return pd.DataFrame(data, columns=['Name', 'Vector', 'Label', 'ProjectType', 'DatasetType'])
        
    def get_annotations(self):

        data = []

        for participant in self.participants:
            for image in participant.Images.values():
                annotations = image.Annotations

                for anno in annotations:
                    data.append([image.FileName, participant.expert, anno.Vector, anno.Label, str(participant.ProjectType), str(participant.DatasetType)])

        return pd.DataFrame(data, columns=['FileName', 'Name', 'Vector', 'Label', 'ProjectType', 'DatasetType'])


    def get_most_active_region(self, image_name: str, radius: int = 1024):

        participants = [participant for participant in self.participants if image_name in participant.Images]

        center_anno = None
        center_count = 0


        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0

        annotations = []
        for participant in participants:
            image = participant.Images[image_name]
            annotations.extend(image.Annotations)

        vectors = [anno.Vector for anno in annotations]

        centers = [(vector[0] + (vector[2] - vector[0]) / 2, vector[1] + (vector[3] - vector[1]) / 2)  for vector in vectors]
        tree = KDTree(centers) 

        for anno in annotations:
            vector = anno.Vector

            center = [(vector[0] + (vector[2] - vector[0]) / 2, vector[1] + (vector[3] - vector[1]) / 2)]
            
            index_per_point = tree.query_radius(center, r=radius)[0]
            count = len(index_per_point)
            
            if count > center_count:
                center_count = count
                
                x_min = max(0, int(vector[0] - radius / 2))
                y_min = max(0, int(vector[1] - radius / 2))
                
                x_max = int(vector[0] + radius / 2)
                y_max = int(vector[1] + radius / 2)
                
                center_anno = anno
        
        return center_anno, center_count, (x_min, y_min, x_max, y_max)



