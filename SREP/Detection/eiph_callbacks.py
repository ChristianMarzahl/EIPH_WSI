from fastai.callbacks import *

from helper.object_detection_helper import *
from helper.nms_center_distance import non_max_suppression_by_distance

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *


class PascalVOCMetricByDistance(Callback):

    def __init__(self, anchors, size, metric_names: list, detect_thresh: float=0.3, nms_thresh: float=0.5
                 , radius: float=25, images_per_batch: int=-1):
        self.ap = 'AP'
        self.anchors = anchors
        self.size = size
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh
        self.radius = radius

        self.images_per_batch = images_per_batch
        self.metric_names_original = metric_names
        self.metric_names = ["{}-{}".format(self.ap, i) for i in metric_names]

        self.evaluator = Evaluator()
        self.boundingBoxes = BoundingBoxes()

    def on_epoch_begin(self, **kwargs):
        self.boundingBoxes.removeAllBoundingBoxes()
        self.imageCounter = 0


    def on_batch_end(self, last_output, last_target, **kwargs):
        bbox_gt_batch, class_gt_batch = last_target
        class_pred_batch, bbox_pred_batch = last_output[:2]

        self.images_per_batch = self.images_per_batch if self.images_per_batch > 0 else class_pred_batch.shape[0]
        for bbox_gt, class_gt, clas_pred, bbox_pred in \
                list(zip(bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch))[: self.images_per_batch]:

            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            if bbox_pred is None:# or len(preds) > 3 * len(bbox_gt):
                continue

            #image = np.zeros((512, 512, 3), np.uint8)
            t_sz = torch.Tensor([(self.size, self.size)])[None].cpu()
            bbox_pred = to_np(rescale_boxes(bbox_pred.cpu(), t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2


            temp_boxes = np.copy(bbox_pred)
            temp_boxes[:, 2] = temp_boxes[:, 0] + temp_boxes[:, 2]
            temp_boxes[:, 3] = temp_boxes[:, 1] + temp_boxes[:, 3]


            to_keep = non_max_suppression_by_distance(temp_boxes, to_np(scores), self.radius, return_ids=True)
            bbox_pred, preds, scores = bbox_pred[to_keep], preds[to_keep].cpu(), scores[to_keep].cpu()

            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0]
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))


            class_gt = to_np(class_gt) - 1
            preds = to_np(preds)
            scores = to_np(scores)

            for box, cla in zip(bbox_gt, class_gt):
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                               w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(self.size,self.size))

                self.boundingBoxes.addBoundingBox(temp)

            # to reduce math complexity take maximal three times the number of gt boxes
            num_boxes = len(bbox_gt) * 3
            for box, cla, scor in list(zip(bbox_pred, preds, scores))[:num_boxes]:
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(self.size, self.size))

                self.boundingBoxes.addBoundingBox(temp)

            #image = self.boundingBoxes.drawAllBoundingBoxes(image, str(self.imageCounter))
            self.imageCounter += 1


    def on_epoch_end(self, last_metrics, **kwargs):
        if self.boundingBoxes.count() > 0:
            self.metrics = {}
            metricsPerClass = self.evaluator.GetPascalVOCMetrics(self.boundingBoxes, IOUThreshold=self.nms_thresh)
            self.metric = max(sum([mc[self.ap] for mc in metricsPerClass]) / len(metricsPerClass), 0)

            for mc in metricsPerClass:
                self.metrics['{}-{}'.format(self.ap, mc['class'])] = max(mc[self.ap], 0)

            return {'last_metrics': last_metrics + [self.metric]}
        else:
            self.metrics = dict(zip(self.metric_names, [0 for i in range(len(self.metric_names))]))
            return {'last_metrics': last_metrics + [0]}