import numpy as np
import copy
from Analysis.image import Image
from Analysis.annotations import Annotation

from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *

class DatasetType(Enum):
    EIPH_Exact = 1
    MitoticFigure = 2
    Asthma = 3
    EIPH_LabelBox = 4

    def __str__(self):
        return self.name.replace("_Exact", "")

class ProjectType(Enum):
    Annotation = 1
    ExpertAlgorithm = 2
    GroundTruth = 3
    
    def __str__(self):
        return self.name

class Expert:

    def __init__(self, participant, bbType, dataset_type: DatasetType, annotation_type: ProjectType):

        expert_lookup = {"3":"1", "4":"2", "5":"3", "6":"4", "7":"5", "8":"6", "9":"7", "11":"8", "12":"9", "13":"10", "19":"19"}
        
        if annotation_type != ProjectType.GroundTruth:
            self.participant = participant.split("_")[0]+ "_" + expert_lookup[participant.split("_")[1]] 
        else:
            self.participant = participant

        self.images = {}
        self.bbType = bbType
        self.dataset_type = dataset_type
        self.annotation_type = annotation_type

    @property
    def Images(self):
        return self.images

    @property
    def ProjectType(self):
        return self.annotation_type

    @property
    def DatasetType(self):
        return self.dataset_type

    @property
    def expert(self):
        return self.participant 

    @property
    def total_annotations(self):
        return sum([len(image.Labels) for image in self.images.values()])

    @property
    def annotations_per_image(self):
        return {image.FileName: len(image.Labels) for image in self.images.values()}

    @property
    def total_seconds_to_label(self):
        return sum([image.seconds_to_label for image in self.images.values() if image.seconds_to_label is not None])

    @property
    def mean_seconds_to_label(self):
        return np.array([image.seconds_to_label for image in self.images.values() if image.seconds_to_label is not None]).mean()

    def calc_metrics(self, second_expert, iou_thresh: float=0.25):

        boundingBoxes = BoundingBoxes()

        for image in self.images.values():
            for bb in image.BB_Boxes:
                boundingBoxes.addBoundingBox(bb)

        for image in second_expert.images.values():
            for bb in image.BB_Boxes:
                boundingBoxes.addBoundingBox(bb)

        evaluator = Evaluator()
        metricsPerClass = evaluator.GetPascalVOCMetrics(boundingBoxes, iou_thresh)

        return metricsPerClass


    def calc_MIoU(self, second_expert, iou_thresh: float=0.5):

        metricsPerClass = self.calc_metrics(second_expert, iou_thresh)

        return np.mean([np.nan_to_num(mc['AP']) for mc in metricsPerClass])

    def calc_sensitivity(self, second_expert, iou_thresh: float=0.25):

        metricsPerClass = self.calc_metrics(second_expert, iou_thresh)

        sensitivity = []
        for mc in metricsPerClass:
            tp = mc['total TP']
            fn = mc['total positives'] -  mc['total TP']

            sensitivity.append(tp / (tp + fn))

        return np.mean(sensitivity)

    def calc_specificity(self, second_expert, fake_cells, iou_thresh: float=0.25):

        metricsPerClass = self.calc_metrics(second_expert, iou_thresh)
        metricsPerClassFakeCells = self.calc_metrics(fake_cells, iou_thresh)

        specificity = []
        for mc, fake in zip(metricsPerClass, metricsPerClassFakeCells):
            fp = mc['total FP']
            tn = fake['total positives'] - fake['total TP']

            specificity.append(tn / (tn + fp))

        return np.mean(specificity)

    def calc_precision(self, second_expert, fake_cells, iou_thresh: float=0.25):
        metricsPerClass = self.calc_metrics(second_expert, iou_thresh)
        metricsPerClassFakeCells = self.calc_metrics(fake_cells, iou_thresh)

        precision = []
        for mc, fake in zip(metricsPerClass, metricsPerClassFakeCells):
            fp = mc['total FP']
            tp = mc['total TP']
            #tn = fake['total positives'] - fake['total TP']

            precision.append(tp / (tp + fp))

        return np.mean(precision)


    def add_experts(self, experts, num_votes:int=2):

        for expert in experts:
            for file_name in expert.Images:
                if file_name not in self.images:
                    self.images[file_name] = copy.deepcopy(expert.Images[file_name])
                    self.images[file_name].bbType = self.bbType
                else:
                    self.images[file_name].annotations += expert.Images[file_name].annotations
            
        for file_name in self.images:
            self.images[file_name].annotations = [copy.deepcopy(anno) for anno in self.images[file_name].annotations]
            for anno in self.images[file_name].annotations:
                anno.anno_type = self.bbType

            self.images[file_name].keep_annotations_with_num_votes(n=num_votes)

    def add_file(self, path : str):

        listOfLines = []

        with open(path, "r") as fileHandler:
            # Get list of all lines in file
            listOfLines = fileHandler.readlines()

        for line in listOfLines:
            splits = line.replace('[','').replace(']','').split('|')

            if len(splits) < 3:
                continue
            
            file_name = splits[0]
            if file_name not in self.images:
                self.images[file_name] = Image(file_name, self.participant, self.bbType)

            self.images[file_name].add_line_information(line)

    def __repr__(self):
        return "{0}:  Annos: {1} Seconds: {2} Type: {3} Project: {4}".format(self.expert, self.total_annotations, self.mean_seconds_to_label, self.dataset_type, self.annotation_type)


