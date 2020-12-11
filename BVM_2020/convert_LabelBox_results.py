import json
import numpy as np
from pathlib import Path
from enum import Enum

from glob import glob
import cv2
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import pandas as pd

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *

from sklearn.neighbors import KDTree

class ProjectType(Enum):
    Annotation = 1
    CrowdAlgorithm = 2
    GroundTruth = 3

class ImageType(Enum):
    NoChanges = 1
    RemovedRects = 2
    NmsArtifacts = 3
    IncreasedGrade = 4

class LabelOrder(Enum):
    AS = 1 # First annotation than supervision
    SA = 2 #


class ExpertResults:

    def __init__(self, file_path):

        self.colors = {0: [255, 255, 0, 255],
                       1: [255, 0, 255, 255],
                       2: [0, 127, 0, 255],
                       3: [255, 127, 0, 255],
                       4: [127, 127, 0, 255]}

        self.images = {}
        self.project_name = ""
        self.created_by = ""

        with open(file_path, 'r') as f:
            images_dict = json.load(f)

            for image in images_dict:
                id = image['External ID']
                if "TestBild" not in id:
                    self.images[id] = LabelBoxImage(image)
                    self.project_name = self.images[id].project_name
                    self.created_by = self.images[id].created_by

    def calc_map(self, gt, iou_thresh: float=0.25, ignore_grade: bool=False):

        boundingBoxes = self._extract_bounding_boxes(gt, ignore_grade)

        evaluator = Evaluator()
        metricsPerClass = evaluator.GetPascalVOCMetrics(boundingBoxes, iou_thresh)
        return np.mean([np.nan_to_num(mc['AP']) for mc in metricsPerClass])

    def drawAllBoundingBoxes(self, gt, image_folder: Path, iou_thresh: float=0.25):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1

        evaluator = Evaluator()
        boundingBoxes = self._extract_bounding_boxes(gt)

        images = {}

        for image_id in gt.images:

            image = np.zeros((376, 256,3), np.uint8)
            image[256:376, 0:256, :] = 255
            image[0:256, 0:256, :] = cv2.imread(str(image_folder/image_id))

            bbxesImage = BoundingBoxes()
            bbxes = boundingBoxes.getBoundingBoxesByImageName(image_id)

            for bb in bbxes:
                bbxesImage.addBoundingBox(bb)

                x1, y1, x2, y2 = bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)

                color = self.colors[bb.getClassId()]
                if bb.getBBType() == BBType.GroundTruth:
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
                    cv2.line(image, (x2, y1), (x1, y2), color, 2)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            metrics_per_class = evaluator.GetPascalVOCMetrics(bbxesImage, iou_thresh)

            for mc in metrics_per_class:
                cv2.putText(image, "Grade: {} mAP: {:01.2f}".format(mc['class'], mc['AP']),
                            (10, 270 + int(20 * mc['class'])),
                            font, fontScale, self.colors[mc['class']], fontThickness, cv2.LINE_AA)

            cv2.putText(image, "mAP: {:01.2f}".format( np.mean([np.nan_to_num(mc['AP']) for mc in metrics_per_class])),
                        (10, 365),
                        font, fontScale, (0,0,0), fontThickness, cv2.LINE_AA)

            images[image_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return images

    def _extract_bounding_boxes(self, gt, ignore_grade: bool=False):
        boundingBoxes = BoundingBoxes()
        # gt boxes
        for image_id in gt.images:

            for x_min, y_min, x_max, y_max, class_id in gt.images[image_id].labels:
                if ignore_grade:
                    class_id = 1

                temp = BoundingBox(imageName=image_id, classId=class_id, x=max(x_min, 0),
                                   y=max(y_min, 0), w=min(256, x_max - x_min), h=min(256, y_max - y_min),
                                   typeCoordinates=CoordinatesType.Absolute,
                                   bbType=BBType.GroundTruth, format=BBFormat.XYWH,
                                   imgSize=(256, 256))
                boundingBoxes.addBoundingBox(temp)

            for x_min, y_min, x_max, y_max, class_id in self.images[image_id].labels:
                if ignore_grade:
                    class_id = 1

                temp = BoundingBox(imageName=image_id, classId=class_id, x=max(x_min, 0),
                                   y=max(y_min, 0), w=min(256, x_max - x_min), h=min(256, y_max - y_min),
                                   typeCoordinates=CoordinatesType.Absolute, classConfidence=1,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(256, 256))
                boundingBoxes.addBoundingBox(temp)
        return boundingBoxes


    def calc_decreased_cells_ration(self, gt, increased_images, iou_thresh=0.5):

        total_cells = 0
        changed_cells = 0

        for image_id in gt.images:
            if image_id in increased_images:

                for x1, y1, x2, y2, class_id_gt in gt.images[image_id].labels:
                    bb1 = (x1, y1, x2, y2)

                    for x_min, y_min, x_max, y_max, class_id in self.images[image_id].labels:
                        bb2 = (x_min, y_min, x_max, y_max)

                        iou = self._iou(bb1, bb2)
                        if iou > iou_thresh:
                            total_cells += 1
                            if class_id == class_id_gt:
                                changed_cells += 1
                            break

        return changed_cells / total_cells * 100

    def calc_removed_cells_ration(self, file_path:str="removed_cells.json", iou_thresh:float=0.5):

        total_cells = 5
        found_cells = 0

        with open(file_path, 'r') as f:
            removed_cells = json.load(f)

            for image_id in removed_cells:
                x1, y1, w, h, _ = removed_cells[image_id]
                bb1 = (x1, y1, x1 + w, y1 + h)

                for x_min, y_min, x_max, y_max, _ in self.images[image_id].labels:
                    bb2 = (x_min, y_min, x_max, y_max)

                    iou = self._iou(bb1, bb2)
                    if iou > iou_thresh:
                        found_cells += 1
                        break

        return found_cells / total_cells * 100

    def confusion_values(self, gt):

        gt_values = []
        predicted_values = []

        for image_id in gt.images:

            boxes = gt.images[image_id].labels
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

            X = np.dstack((center_x, center_y))[0]
            tree = KDTree(X)

            for bb2 in self.images[image_id].labels:
                radius = (bb2[2] - bb2[0]) * 0.5

                query_center_x = bb2[0] + (bb2[2] - bb2[0]) / 2
                query_center_y = bb2[1] + (bb2[3] - bb2[1]) / 2

                query_x = np.dstack(([query_center_x], [query_center_y]))[0]
                ind = tree.query_radius(query_x, r=radius)[0]
                if len(ind) > 0:
                    gt_values.append(boxes[ind][0, 4])
                    predicted_values.append(bb2[4])


        return gt_values, predicted_values

    def __str__(self):
        return self.project_name + " " + self.created_by

    @property
    def seconds(self):
        return [img.seconds_to_label for img in self.images.values()]

    @property
    def file_names(self):
        return [img for img in self.images]

    @property
    def grade(self):
        return [np.mean(img.labels[:, 4]) for img in self.images.values()]

    @property
    def project_type(self):
        return ProjectType.CrowdAlgorithm if "Labels" in self.project_name else ProjectType.Annotation

    def _iou(self, bb1, bb2):
        x1, y1, x2, y2 = bb1
        x_min, y_min, x_max, y_max = bb2

        # determine the coordinates of the intersection rectangle
        x_left = max(x1, x_min)
        y_top = max(y1, y_min)
        x_right = min(x2, x_max)
        y_bottom = min(y2, y_max)

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (x2 - x1) * (y2 - y1)
        bb2_area = (x_max - x_min) * (y_max - y_min)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

        return iou


class LabelBoxImage:

    def __init__(self, json_object, offset: int = 2):

        self.created_by = json_object['Created By']
        self.seconds_to_label = json_object['Seconds to Label']
        self.external_id = json_object['External ID']
        self.dataset_name = json_object['Dataset Name']
        self.project_name = json_object['Project Name']
        self.created_by = json_object['Created By']
        labels_dict = json_object['Label']

        self.labels = []
        for class_id in labels_dict:
            for geo in labels_dict[class_id]:
                x_min, y_min = geo['geometry'][0]['x'], geo['geometry'][0]['y']
                x_max, y_max = geo['geometry'][2]['x'], geo['geometry'][2]['y']

                x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                y_min, y_max = min(y_min, y_max), max(y_min, y_max)

                if x_min > offset and x_max < 256 - offset and y_min > offset and y_max < 256 - offset:
                    self.labels.append([x_min, y_min, x_max, y_max, int(class_id)])

        self.labels = np.array(self.labels)

        self.increase_class = ['01_EIPH_563479 Turnbull blue.png', '02_EIPH_574162 Turnbull blue-001.png', \
                                '04_EIPH_567017 Turnbull blue-001.png', '08_EIPH_574999 Berliner Blau.png', \
                               '13_EIPH_570370 Berliner Blau.png']

        self.images_with_artifacts = []

        self.images_with_removed_cells = ["30_EIPH_568355 Turnbull blue.png", "23_EIPH_563476 Turnbull blue.png",
                                      "17_EIPH_575796 Turnbull blue.png", "14_EIPH_568381 berliner blau-001.png",
                                      "28_EIPH_569948 L berliner blau.png"]


    def calc_changes(self, gt_image):

        gt_num_cells = len(gt_image.labels)
        num_cells = len(self.labels)

        unchanged_boxes = 0
        changed_grade = 0

        self_boxes = np.copy(self.labels)
        for gt_x_min, gt_y_min, gt_x_max, gt_y_max, gt_grade in gt_image.labels:
            for id, row in enumerate(self_boxes):
                x_min, y_min, x_max, y_max, grade = row

                if x_min == gt_x_min and y_min == gt_y_min and x_max == gt_x_max and y_max == gt_y_max:
                    if gt_grade == grade:
                        unchanged_boxes += 1
                    else:
                        changed_grade += 1

                    self_boxes = np.delete(self_boxes, id, axis=0)
                    break
        changed_boxes = len(self_boxes)

        return gt_num_cells, num_cells, unchanged_boxes, changed_grade, changed_boxes


    def confusion_values(self, gt_image):

        gt_values = []
        predicted_values = []

        boxes = gt_image.labels
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

        tree = KDTree(np.dstack((center_x, center_y))[0])

        for bb2 in self.labels:
            radius = (bb2[2] - bb2[0]) * 0.5

            query_center_x = bb2[0] + (bb2[2] - bb2[0]) / 2
            query_center_y = bb2[1] + (bb2[3] - bb2[1]) / 2

            query_x = np.dstack(([query_center_x], [query_center_y]))[0]
            ind = tree.query_radius(query_x, r=radius)[0]
            if len(ind) > 0:
                gt_values.append(boxes[ind][0, 4])
                predicted_values.append(bb2[4])

        return gt_values, predicted_values


    @property
    def image_type(self):

        if self.external_id in self.increase_class:
            return ImageType.IncreasedGrade

        if self.external_id in self.images_with_removed_cells:
            return ImageType.RemovedRects

        if self.external_id in self.images_with_artifacts:
            return ImageType.NmsArtifacts

        return ImageType.NoChanges


class EIPH_Statistics:


    def __init__(self, path: Path, ground_truth_name: str = "GroundTruth"):

        results_experts = {}
        self.ground_truth = None
        for file in glob(str(path / '*.json')):

            if ground_truth_name in file:
                self.ground_truth = ExpertResults(file)
            else:
                temp = ExpertResults(file)
                initials = temp.project_name.split('-')[0]
                if initials not in results_experts:
                    results_experts[initials] = {ProjectType.Annotation: None, ProjectType.CrowdAlgorithm: None}

                results_experts[initials][temp.project_type] = temp

        self.results_experts = results_experts

        self.label_order = {
            "CB": LabelOrder.AS,
            "CM": LabelOrder.AS,
            "JA": LabelOrder.SA,
            "MA": LabelOrder.AS,
            "MH": LabelOrder.AS,
            "PM": LabelOrder.AS,
            "RK": LabelOrder.SA,
            "SC": LabelOrder.SA,
            "SE": LabelOrder.SA,
            "SG": LabelOrder.SA,
            "DSS": LabelOrder.AS,
            "JM": LabelOrder.AS,
            "MB": LabelOrder.SA,
            "LJ": LabelOrder.AS,
            "LM": LabelOrder.SA,
            "SO": LabelOrder.AS,
            "MS": LabelOrder.AS,
            "PA": LabelOrder.SA,
            "HJ": LabelOrder.SA,
            "DL": LabelOrder.AS,
            "JS": LabelOrder.AS,
            "KJ": LabelOrder.AS,
            "BT": LabelOrder.AS,
        }

        self.skill_level = {
            "CB": 4,
            "CM": 3,
            "JA": 1,
            "MA": 2,
            "MH": 3,
            "PM": 2,
            "RK": 4,
            "SC": 1,
            "SE": 2,
            "SG": 2,
            "HJ": 2,
            "DSS": 3,
            "JM": 1,
            "SO": 1,
            "LM": 3,
            "MB": 3,
            "LJ": 3,
            "PA": 4,
            "MS": 4,
            "JS": 1,
            "KJ": 2,
            "BT": 4,
            "DL": "DL"
        }

        self.increase_class = ['01_EIPH_563479 Turnbull blue.png', '02_EIPH_574162 Turnbull blue-001.png', \
                                '04_EIPH_567017 Turnbull blue-001.png', '08_EIPH_574999 Berliner Blau.png', \
                               '13_EIPH_570370 Berliner Blau.png']

        self.images_with_artifacts = [] #24; 25; 23; 19; 08

        self.images_with_removed_cells = ["30_EIPH_568355 Turnbull blue.png", "23_EIPH_563476 Turnbull blue.png",
                                      "17_EIPH_575796 Turnbull blue.png", "14_EIPH_568381 berliner blau-001.png",
                                      "28_EIPH_569948 L berliner blau.png"]

    def calc_statistics_image(self):

        data = []

        for key in self.results_experts:
            initials = self.results_experts[key]

            labels = initials[ProjectType.CrowdAlgorithm]
            no_labels = initials[ProjectType.Annotation]


            if labels is not None:
                for image in labels.images.values():
                    gt_num_cells, num_cells, unchanged_boxes, changed_grade, changed_boxes = \
                        image.calc_changes(self.ground_truth.images[image.external_id])
                    image_name, seconds, grade, grade_gt, image_type, acc = self._extract_image_statistics(image)

                    data.append([key, self.skill_level[key], image_name, ProjectType.CrowdAlgorithm, self.label_order[key], seconds, grade, grade_gt,
                                 image_type, acc, gt_num_cells, num_cells, unchanged_boxes, changed_grade, changed_boxes])

            if no_labels is not None:
                for image in no_labels.images.values():
                    gt_num_cells, num_cells, unchanged_boxes, changed_grade, changed_boxes = \
                        image.calc_changes(self.ground_truth.images[image.external_id])

                    image_name, seconds, grade, grade_gt, image_type, acc = self._extract_image_statistics(image)

                    data.append([key, self.skill_level[key], image_name, ProjectType.Annotation, self.label_order[key], seconds, grade, grade_gt,
                                 image_type, acc, gt_num_cells, num_cells, unchanged_boxes, changed_grade, changed_boxes])

        return pd.DataFrame(data, columns=['Initials', 'Skill', 'File', 'Type', 'Order', 'Time', 'Grade',
                                           'Grade GT', 'ImageType', 'Acc', 'gt_num_cells', 'num_cells',
                                           'unchanged_boxes', 'changed_grade', 'changed_boxes'])

    def calc_fleiss_kappa(self, project_type:ProjectType = ProjectType.CrowdAlgorithm):

        images_dict = {}

        for key in self.ground_truth.images:
            images_dict[key] = []


        for key in self.results_experts:
            initials = self.results_experts[key]

            labels_type= initials[project_type]
            labels_AS = initials[ProjectType.Annotation]

            if labels_type is not None:
                for key in images_dict:
                    images_dict[key].append(labels_type.images[key].labels)

        data = []
        cell_id = 0

        for image_key in self.ground_truth.images:

            for bb2 in self.ground_truth.images[image_key].labels:

                row = [image_key+"_{}".format(cell_id), 0, 0, 0, 0, 0, 0]

                radius = (bb2[2] - bb2[0]) * 0.5

                query_center_x = bb2[0] + (bb2[2] - bb2[0]) / 2
                query_center_y = bb2[1] + (bb2[3] - bb2[1]) / 2

                query_x = np.dstack(([query_center_x], [query_center_y]))[0]

                for image_boxes in images_dict[image_key]:

                    boxes = np.copy(image_boxes)

                    center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
                    center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

                    X = np.dstack((center_x, center_y))[0]
                    tree = KDTree(X)

                    ind = tree.query_radius(query_x, r=radius)[0]
                    if len(ind) > 0:
                        index = boxes[ind, 4][0] + 1
                        row[index] += 1

                cell_id += 1
                data.append(row)

        fleiss_kappa_df =  pd.DataFrame(data, columns = ['Name', 0, 1, 2, 3, 4, 'Pi'])

        # https://en.wikipedia.org/wiki/Fleiss%27_kappa
        total_votes = 0
        for i, trial in fleiss_kappa_df.iterrows():
            num_votes = sum([trial[t] for t in range(5)])
            total_votes += num_votes
            if num_votes > 2:
                fleiss_kappa_df.loc[i, 'Pi'] = (1 / (num_votes * (num_votes - 1))) * (
                        sum([pow(trial[t], 2) for t in range(5)]) - num_votes)

        pj = [sum(fleiss_kappa_df[i].tolist()) / total_votes for i in range(5)]
        P = np.mean(fleiss_kappa_df['Pi'].tolist())
        Pe = sum([pow(p, 2) for p in pj])
        K = (P - Pe) / (1 - Pe)

        return K

    def calc_statistics_initials(self):

        data = []

        for key in self.results_experts:
            initials = self.results_experts[key]

            labels = initials[ProjectType.CrowdAlgorithm]
            no_labels = initials[ProjectType.Annotation]

            if labels is not None:
                row = self.extract_expert_metrics(key, labels)
                data.append(list(row.values()) + [ProjectType.CrowdAlgorithm])

            if no_labels is not None:
                row = self.extract_expert_metrics(key, no_labels)
                data.append(list(row.values()) + [ProjectType.Annotation])

        columns = list(row.keys()) + ["Type"]

        return pd.DataFrame(data, columns=columns)

    def extract_expert_metrics(self, key, expert):
        row = {'Initials': key, "Skill": self.skill_level[key], "mAP": np.median(expert.calc_map(self.ground_truth)),
               "mAP-IG": np.median(expert.calc_map(self.ground_truth, ignore_grade=True)), "Acc": np.nan,
               "ratio_decreased_cells": np.nan, "ratio_removed_boxes": np.nan, "seconds": np.nan, "grade": np.nan,
               "num_cells": np.nan, "num_interactionen": np.nan, "Order": self.label_order[key]}

        temp_get, temp_pred = expert.confusion_values(self.ground_truth)
        row["Acc"] = accuracy_score(temp_get, temp_pred)
        row[
            "ratio_decreased_cells"] = expert.calc_decreased_cells_ration(self.ground_truth, self.increase_class)
        row["ratio_removed_boxes"] = expert.calc_removed_cells_ration("removed_cells.json")
        row["seconds"] = np.median(expert.seconds)
        row["grade"] = np.mean(expert.grade)
        row["num_cells"] = self._calc_number_of_interactions(expert)
        return row

    def calc_statistics_intra_observer(self):

        data = []

        for key in self.results_experts:
            initials = self.results_experts[key]

            labels = initials[ProjectType.CrowdAlgorithm]
            no_labels = initials[ProjectType.Annotation]

            if labels is not None and no_labels is not None:
                gt_values, predicted_values = labels.confusion_values(no_labels)
                kappa = cohen_kappa_score(gt_values, predicted_values)
                acc = accuracy_score(gt_values, predicted_values)
                map = labels.calc_map(no_labels)
                map_ig = labels.calc_map(no_labels, ignore_grade=True)
                grade_error = np.mean(labels.grade) - np.mean(no_labels.grade)

                data.append([key, self.skill_level[key], self.label_order[key], grade_error, kappa,
                             map, map_ig, acc])

        return pd.DataFrame(data, columns=['Initials', 'Skill', 'Order', 'Grade-Error',
                                           'Kappa', 'mAP', 'mAP-IG', 'Acc'])

    def calc_confusion_matrix(self, project_type:ProjectType = ProjectType.Annotation):

        gt_values = []
        predicted_values = []

        for key in self.results_experts:
            initials = self.results_experts[key]

            expert = initials[project_type]

            if expert is not None:
                temp_get, temp_pred = expert.confusion_values(self.ground_truth)

                gt_values += temp_get
                predicted_values += temp_pred

        return confusion_matrix(gt_values, predicted_values)

    def _extract_image_statistics(self, image: LabelBoxImage):

        image_name = image.external_id
        seconds = image.seconds_to_label
        grade = np.mean(image.labels[:, 4])
        image_type = image.image_type

        grade_gt = np.mean(self.ground_truth.images[image_name].labels[:, 4])

        gt_values, predicted_values = image.confusion_values(self.ground_truth.images[image_name])
        acc = accuracy_score(gt_values, predicted_values)

        return [image_name, seconds, grade, grade_gt, image_type, acc]

    def _calc_number_of_interactions(self, expert: ExpertResults):

        return  sum([len(image.labels) for image in expert.images.values()])







'''
path = Path("Results")
statistics = EIPH_Statistics(path)
k1 = statistics.calc_fleiss_kappa(ProjectType.Supervision)
k2 = statistics.calc_fleiss_kappa(ProjectType.Annotation)

print()


temp = statistics.calc_statistics_image()
temp = statistics.calc_statistics_intra_observer()
temp = statistics.calc_statistics_initials()



images_folder = Path("/server/born_pix_cm/EIPH_WSI/Studie/")

file_path = 'Results/SG-EIPH-Labels.json'
temp = ExpertResults(file_path)

file_path = 'Results/GroundTruth.json'
gt = ExpertResults(file_path)


temp_get, temp_pred = temp.confusion_values(gt)
print(len(temp_get))


cm = confusion_matrix(temp_get, temp_pred)
cmap = plt.cm.Blues
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4],
       title="No Labels",
       ylabel="Actual",
       xlabel="Predicted")

plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = 'd'  # '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

print()

from pathlib import Path
from glob import glob

path = Path("Results")

results_experts = {}
ground_truth = None
for file in glob(str(path / '*.json')):

    if "GroundTruth" in file:
        ground_truth = ExpertResults(file)
    else:
        temp = ExpertResults(file)
        results_experts[temp.project_name] = temp

label_time = [np.median(expert.seconds) for expert in results_experts.values() if expert.project_type == ProjectType.NoLabels]
label_time_annotations = [np.median(expert.seconds) for expert in results_experts.values() if expert.project_type == ProjectType.Labels]


print()

'''
