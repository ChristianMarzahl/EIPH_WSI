from enum import Enum
import numpy as np
import pickle
from pathlib import Path
import cv2
from PIL import Image
from openslide import OpenSlide
from probreg import cpd
from probreg import transformation as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import itertools

from sklearn.neighbors import LocalOutlierFactor


class NodeOrientation(Enum):
    TOP         = 0
    NORTH_WEST  = 1
    NORTH_EAST  = 2
    SOUTH_WEST  = 3
    SOUTH_EAST  = 4


class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object.

    """

    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)


class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def create(cls, x, y, w, h):

        cx, cy = x + w // 2, y + h // 2

        return cls(cx, cy, w, h)


    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <  self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, source_boundary, source_slide, target_boundary, target_slide, 
                        depth=0, target_depth=4, thumbnail_size=(2048, 2048),
                        run_async=False, node_orientation:NodeOrientation = NodeOrientation.TOP,
                        parent=None, homography:bool=True, filter_outliner:bool=False, **kwargs):


        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """
        self.kwargs = kwargs
        self.parent = parent
        self.node_orientation = node_orientation
        self.run_async = run_async
        self.thumbnail_size = thumbnail_size
        self.depth = depth
        self.target_depth = target_depth 

        self.source_slide_name = source_slide._filename
        self.source_boundary = source_boundary
        self._source_slide = source_slide
        self.source_thumbnail, self.source_scale = self.get_region_thumbnail(source_slide, self.source_boundary, self.thumbnail_size)

        self.target_slide_name = target_slide._filename
        self.target_boundary = target_boundary
        self._target_slide = target_slide       
        self.target_thumbnail, self.target_scale = self.get_region_thumbnail(target_slide, self.target_boundary, self.thumbnail_size)

        self.ptsA, self.ptsB, self.matchedVis = self.extract_matching_points_old(self.source_thumbnail, self.target_thumbnail, source_scale=self.source_scale, target_scale=self.target_scale, **kwargs)

        if filter_outliner:
            self.ptsA, self.ptsB, self.scale_factors = self.filter_outliner(self.ptsA, self.ptsB)
        
        if homography:
            homography, mask = cv2.findHomography(self.ptsA, self.ptsB, cv2.RANSAC) 
            self.tf_param = tf.AffineTransformation(homography[:2, :2], homography[:2, 2:].reshape(-1))
            self.sigma2, self.q = -1, -1
        else:
            self.tf_param, self.sigma2, self.q = cpd.registration_cpd(self.ptsA, self.ptsB, 'affine')

        self.b = self.tf_param.b
        self.t = self.tf_param.t
        self.mpp_x_scale = self.tf_param.b[0][0]
        self.mpp_y_scale = self.tf_param.b[1][1]

        self.max_points = len(self.ptsA)
        self.points = self.ptsA
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

        self.nw, self.ne, self.se, self.sw = None, None, None, None
        if depth < target_depth:
            self.divide()

    @property
    def source_thumbnail(self):

        if self._source_thumbnail is None:
            self.source_thumbnail, self.source_scale = self.get_region_thumbnail(self.source_slide, self.source_boundary, self.thumbnail_size)

        return self._source_thumbnail

    @source_thumbnail.setter 
    def source_thumbnail(self, thumbnail):
        self._source_thumbnail = thumbnail

    @property
    def target_thumbnail(self):

        if self._target_thumbnail is None:
            self._target_thumbnail, self.target_scale = self.get_region_thumbnail(self.target_slide, self.target_boundary, self.thumbnail_size)

        return self._target_thumbnail

    @target_thumbnail.setter 
    def target_thumbnail(self, thumbnail):
        self._target_thumbnail = thumbnail

    @property
    def source_slide(self):

        if self._source_slide is None:
            raise Exception('Please set source_slide and target_slide after pickle load')

        return self._source_slide

    @source_slide.setter 
    def source_slide(self, slide):

        self.source_slide_name = slide._filename
        self._source_slide = slide

        if self.nw is not None: self.nw.source_slide = slide
        if self.ne is not None: self.ne.source_slide = slide
        if self.se is not None: self.se.source_slide = slide
        if self.sw is not None: self.sw.source_slide = slide

    @property
    def target_slide(self):

        if self._target_slide is None:
            raise Exception('Please set source_slide and target_slide after pickle load')

        return self._target_slide

    @target_slide.setter 
    def target_slide(self, slide):

        self.target_slide_name = slide._filename
        self._target_slide = slide

        if self.nw is not None: self.nw.target_slide = slide
        if self.ne is not None: self.ne.target_slide = slide
        if self.se is not None: self.se.target_slide = slide
        if self.sw is not None: self.sw.target_slide = slide

    @property
    def source_name(self):
        return Path(self.source_slide_name).stem
            
    @property
    def target_name(self):
        return Path(self.target_slide_name).stem
        
    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        
        s = ""
        if self.depth == 0:
            s +=  f"Source: {self.source_name} \n"
            s +=  f"Target: {self.target_name} \n"
        
        s += f"Source: {self.source_boundary} Target: {self.target_boundary}" + '\n' 
        s += sp + f'x: [{self.tf_param.b[0][0]:4.3f}, {self.tf_param.b[0][1]:4.3f}, {self.tf_param.t[0]:4.3f}], y: [{self.tf_param.b[1][0]:4.3f}, {self.tf_param.b[1][1]:4.3f}, {self.tf_param.t[1]:4.3f}]] q: {self.q:4.3f}'
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        source_cx, source_cy = self.source_boundary.cx, self.source_boundary.cy
        source_w, source_h = self.source_boundary.w / 2, self.source_boundary.h / 2

        # transform target bounding box
        #new_xmin, new_ymin = self.tf_param.transform((self.source_boundary.west_edge, self.source_boundary.north_edge))
        #new_xmax, new_ymax = self.tf_param.transform((self.source_boundary.east_edge, self.source_boundary.south_edge))

        new_xmin, new_ymin = self.transform_boxes([(self.source_boundary.west_edge, self.source_boundary.north_edge, 50, 50)])[0][:2]
        new_xmax, new_ymax = self.transform_boxes([(self.source_boundary.east_edge, self.source_boundary.south_edge, 50, 50)])[0][:2]

        # set new box coordinates withhin old limits
        #new_ymin, new_xmin = max(new_ymin, self.target_boundary.north_edge), max(new_xmin, self.target_boundary.west_edge)
        #new_ymax, new_xmax = min(new_ymax, self.target_boundary.south_edge), min(new_xmax, self.target_boundary.east_edge)

        new_ymin, new_xmin = max(new_ymin, 0), max(new_xmin, 0)
        new_ymax, new_xmax = min(new_ymax, self.target_slide.dimensions[1]), min(new_xmax, self.target_slide.dimensions[0])

        # transform target center
        #target_cx, target_cy = self.tf_param.transform((self.source_boundary.cx, self.source_boundary.cy))
        target_cx, target_cy = self.transform_boxes([(self.source_boundary.cx, self.source_boundary.cy, 50, 50)])[0][:2]

        # create target boxes
        target_nw = Rect.create(Rect, new_xmin, new_ymin, target_cx - new_xmin, target_cy - new_ymin)
        target_sw = Rect.create(Rect, new_xmin, target_cy, target_cx - new_xmin, new_ymax - target_cy)

        target_ne = Rect.create(Rect, target_cx, new_ymin,  new_xmax - target_cx, target_cy - new_ymin)
        target_se = Rect.create(Rect, target_cx, target_cy, new_xmax - target_cx, new_ymax - target_cy)

        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        if self.run_async == False:
            self.nw = QuadTree(Rect(source_cx - source_w/2, source_cy - source_h/2, source_w, source_h), self.source_slide, 
                                        target_nw, self.target_slide, 
                                        depth=self.depth + 1, target_depth=self.target_depth, 
                                        run_async=self.run_async, node_orientation=NodeOrientation.NORTH_WEST, parent=self, **self.kwargs)
            self.ne = QuadTree(Rect(source_cx + source_w/2, source_cy - source_h/2, source_w, source_h), self.source_slide,
                                        target_ne, self.target_slide,
                                        depth=self.depth + 1, target_depth=self.target_depth,  
                                        run_async=self.run_async, node_orientation=NodeOrientation.NORTH_EAST, parent=self, **self.kwargs)
            self.se = QuadTree(Rect(source_cx + source_w/2, source_cy + source_h/2, source_w, source_h), self.source_slide,
                                        target_se, self.target_slide,
                                        depth=self.depth + 1, target_depth=self.target_depth,  
                                        run_async=self.run_async, node_orientation=NodeOrientation.SOUTH_WEST, parent=self, **self.kwargs)
            self.sw = QuadTree(Rect(source_cx - source_w/2, source_cy + source_h/2, source_w, source_h), self.source_slide,
                                        target_sw, self.target_slide,
                                        depth=self.depth + 1, target_depth=self.target_depth, 
                                        run_async=self.run_async, node_orientation=NodeOrientation.SOUTH_EAST, parent=self, **self.kwargs)
        else:
            with Pool(4) as pool:
                # , self.ne, self.se, self.sw 
                self.nw, self.ne, self.se, self.sw = zip(*pool.map(QuadTree, [[Rect(source_cx - source_w/2, source_cy - source_h/2, source_w, source_h), 
                                        self.source_slide, target_nw, self.target_slide, 
                                        self.depth + 1, self.target_depth, self.thumbnail_size, self.run_async],
                                        
                                        [Rect(source_cx + source_w/2, source_cy - source_h/2, source_w, source_h), 
                                        self.source_slide, target_ne, self.target_slide,
                                        self.depth + 1, self.target_depth, self.thumbnail_size, self.run_async], 
                                        
                                        [Rect(source_cx + source_w/2, source_cy + source_h/2, source_w, source_h), 
                                        self.source_slide, target_se, self.target_slide,
                                        self.depth + 1, self.target_depth, self.thumbnail_size, self.run_async], 
                                        
                                        [Rect(source_cx - source_w/2, source_cy + source_h/2, source_w, source_h), 
                                        self.source_slide, target_sw, self.target_slide,
                                        self.depth + 1, self.target_depth, self.thumbnail_size, self.run_async], 
                                        ]))
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.source_boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints
    
    def draw_feature_points(self, num_sub_pic:int=5, figsize=(16, 16)):
        
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        fig.suptitle(f'{self.source_name} --> {self.target_name}')
        gs = fig.add_gridspec(5, num_sub_pic)

        f_ax_match = fig.add_subplot(gs[:2, :])
        f_ax_match.imshow(self.matchedVis)

        tf_temp = tf.AffineTransformation(self.tf_param.b, self.tf_param.t)
        
        for idx, (pA, pB) in enumerate(zip(self.ptsA[:num_sub_pic].copy(), self.ptsB[:num_sub_pic].copy())):
            size = 512

            transformed = pA.copy()
            pA = (pA + (self.source_boundary.west_edge, self.source_boundary.north_edge)).astype(int)
            pB = (pB + (self.target_boundary.west_edge, self.target_boundary.north_edge)).astype(int)

            transformed = (tf_temp.transform(transformed) + (self.target_boundary.west_edge, self.target_boundary.north_edge)).astype(int)
            pA = pA.astype(int)

            size_target_x, size_target_y = int(size * self.mpp_x_scale), int(size * self.mpp_y_scale)


            image_source = self.source_slide.read_region(location=pA.astype(int) - (size // 2, size // 2), 
                                                        level=0, size=(size, size))
            image_target = self.target_slide.read_region(location=pB.astype(int) - (size_target_x // 2, size_target_y // 2), 
                                                        level=0, size=(size_target_x, size_target_y))
            image_target_trans = self.target_slide.read_region(location=transformed - (size_target_x // 2, size_target_y // 2), 
                                                        level=0, size=(size_target_x, size_target_y))


            ax = fig.add_subplot(gs[2, idx])
            ax.set_title(f'Source:  {pA}')
            ax.imshow(image_source)

            ax = fig.add_subplot(gs[3, idx])
            ax.set_title(f'Trans:  {transformed}')
            ax.imshow(image_target_trans)

            ax = fig.add_subplot(gs[4, idx])
            ax.set_title(f'GT:  {pB}')
            ax.imshow(image_target)
        
        if self.divided:
            return fig, [self.nw.draw_feature_points(num_sub_pic, figsize),
                            self.ne.draw_feature_points(num_sub_pic, figsize),
                            self.se.draw_feature_points(num_sub_pic, figsize),
                            self.sw.draw_feature_points(num_sub_pic, figsize)]
        else:
            return fig, None
        
    def filter_boxes(self, boxes):
        """[summary]
            Filter boxes that are not visibile in the quadtree level 
        Args:
            boxes ([type]): [Array of boxes: [xc, cy, w, h]]

        Returns:
            [type]: [description]
        """

        boxes = boxes[((boxes[:, 0] > self.source_boundary.west_edge) & (boxes[:, 0] < self.source_boundary.east_edge))]
        boxes = boxes[((boxes[:, 1] > self.source_boundary.north_edge) & (boxes[:, 1] < self.source_boundary.south_edge))]

        return boxes

    def transform_boxes(self, boxes):
        """[summary]
            Transform box coordinages from the soure to the target domain coordinate system
        Args:
            boxes ([type]): [Array of boxes: [xc, cy, w, h]]

        Returns:
            [type]: [description]
        """

        tf_temp = tf.AffineTransformation(self.tf_param.b, self.tf_param.t)

        result_boxes = []
        for box in boxes:
            box = np.array(box)
            point = Point(box[0], box[1])
            
            if self.nw is not None and self.nw.sigma2 < self.sigma2 and self.nw.source_boundary.contains(point): #q
                box = self.nw.transform_boxes([box])[0]   
            elif self.ne is not None and self.ne.sigma2 < self.sigma2 and self.ne.source_boundary.contains(point):
                box = self.ne.transform_boxes([box])[0] 
            elif self.se is not None and self.se.sigma2 < self.sigma2 and self.se.source_boundary.contains(point):
                box = self.se.transform_boxes([box])[0] 
            elif self.sw is not None and self.sw.sigma2 < self.sigma2 and self.sw.source_boundary.contains(point):
                box = self.sw.transform_boxes([box])[0] 
            else:

                source_boxes = box[:2] - (self.source_boundary.west_edge, self.source_boundary.north_edge) 
                transformed_xy = tf_temp.transform(source_boxes) + (self.target_boundary.west_edge, self.target_boundary.north_edge)
                transformed_wh = box[2:] * np.array([self.mpp_x_scale, self.mpp_y_scale])

                box = np.hstack([transformed_xy, transformed_wh])

            result_boxes.append(box)

        #if self.depth == 0:
            #result_boxes = np.array(list(itertools.chain(*result_boxes)))

        return result_boxes
        

    def draw_annotations(self, boxes, figsize=(16, 16), num_sub_pic:int=5):
        """[summary]
        Draw annotations on patches from the source and target slide
        Args:
            boxes ([type]): Array of boxes: [xc, cy, w, h]]
            figsize (tuple, optional): description. Defaults to (16, 16).
            num_sub_pic (int, optional): description. Defaults to 5.

        Returns:
            [type]: Array of figures for each level
        """

        source_boxes = boxes.copy()
        source_boxes = self.filter_boxes(source_boxes)
        target_boxes = self.transform_boxes(source_boxes)
        
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        fig.suptitle(f'{self.source_name} --> {self.target_name}')
        gs = fig.add_gridspec(4, num_sub_pic)

        f_ax_match = fig.add_subplot(gs[:2, :])
        f_ax_match.imshow(self.matchedVis)
        
        for idx, (source_box, target_box)  in enumerate(zip(source_boxes[:num_sub_pic], target_boxes[:num_sub_pic])):
            size = 512

            pA = np.array(source_box[:2]).astype(int)
            source_anno_width, source_anno_height = source_box[2:4]
            source_x1, source_y1 = (size / 2) - source_anno_width / 2, (size / 2)  - source_anno_height / 2
            
            
            transformed = target_box[:2].astype(int)
            size_target_x, size_target_y = abs(int(size * self.mpp_x_scale)), abs(int(size * self.mpp_y_scale))
            
            target_anno_width, target_anno_height = int(source_anno_width * self.mpp_x_scale), int(source_anno_height * self.mpp_y_scale)
            target_x1, target_y1 = (size_target_x / 2) - target_anno_width / 2, (size_target_y / 2)  - target_anno_height / 2
            

            image_source = self.source_slide.read_region(location=pA.astype(int) - (size // 2, size // 2), 
                                                        level=0, size=(size, size))
            image_target_trans = self.target_slide.read_region(location=transformed - (size_target_x // 2, size_target_y // 2), 
                                                        level=0, size=(size_target_x, size_target_y))


            ax = fig.add_subplot(gs[2, idx])
            ax.set_title(f'Source:  {pA}')
            ax.imshow(image_source)            
            rect = patches.Rectangle((source_x1, source_y1), source_anno_width, source_anno_height, 
                                linewidth=3, edgecolor='m', facecolor='none')
            ax.add_patch(rect)
            

            ax = fig.add_subplot(gs[3, idx])
            ax.set_title(f'Trans:  {transformed}')
            ax.imshow(image_target_trans)
            rect = patches.Rectangle((target_x1, target_y1), target_anno_width, target_anno_height, 
                                 linewidth=3, edgecolor='m', facecolor='none')
            ax.add_patch(rect)

        
        if self.divided:
            return fig, [self.nw.draw_annotations(boxes, figsize, num_sub_pic),
                            self.ne.draw_annotations(boxes, figsize, num_sub_pic),
                            self.se.draw_annotations(boxes, figsize, num_sub_pic),
                            self.sw.draw_annotations(boxes, figsize, num_sub_pic)]
        else:
            return fig, None

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.source_boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)

            
    def filter_outliner(self, ptsA, ptsB):
        
        scales =  ptsA / ptsB
        
        #if self.parent is None:
        inliners = LocalOutlierFactor(n_neighbors=int(len(scales) * 0.25)).fit_predict(scales) == 1
        #else:
        #    inliners = scales[:, 0] > min(self.parent.scale_factors[:, 0]) & scales[:, 0] < max(self.parent.scale_factors[:, 0]) & scales[:, 1] > min(self.parent.scale_factors[:, 1]) & scales[:, 1] < max(self.parent.scale_factors[:, 1])

        return ptsA[inliners], ptsB[inliners], ptsA[inliners] / ptsB[inliners]
        
            
    def extract_matching_points(self, source_image, target_image, maxFeatures:int=500, 
                    keepPercent:float=0.2, debug=False, 
                    source_scale:[tuple]=[(1,1)], target_scale:[tuple]=[(1,1)]):

        source_scale = np.array(source_scale)
        target_scale = np.array(target_scale)

        source_image = np.array(source_image) if type(source_image) == Image.Image else source_image
        target_image = np.array(target_image) if type(target_image) == Image.Image else target_image
        
        # convert both the input image and template to grayscale
        imageGray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) 
        templateGray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        

        kpsA, kpsB, matches = [], [], []
        # extract features from the four regions top right, top left ...
        for region in       [
                             NodeOrientation.TOP,
                             #NodeOrientation.NORTH_WEST, 
                             #NodeOrientation.NORTH_EAST,
                             #NodeOrientation.SOUTH_WEST, 
                             #NodeOrientation.SOUTH_EAST
                            ]:

            # set all non active regions to zero
            region_image = np.zeros_like(imageGray)
            region_template = np.zeros_like(templateGray)

            y_center_region, x_center_region =      np.array(region_image.shape[:2]) // 2
            y_center_template, x_center_template =  np.array(region_template.shape[:2]) // 2

            if region == NodeOrientation.TOP:
                region_image = imageGray
                region_template = templateGray

            if region == NodeOrientation.NORTH_WEST:
                region_image[:y_center_region, :x_center_region] = imageGray[:y_center_region, :x_center_region]
                region_template[:y_center_template, :x_center_template] = templateGray[:y_center_template, :x_center_template]

            if region == NodeOrientation.NORTH_EAST:
                region_image[:y_center_region, x_center_region:] = imageGray[:y_center_region, x_center_region:]
                region_template[:y_center_template, x_center_template:] = templateGray[:y_center_template, x_center_template:]

            if region == NodeOrientation.SOUTH_WEST:
                region_image[y_center_region:, :x_center_region] = imageGray[y_center_region:, :x_center_region]
                region_template[y_center_template:, :x_center_template] = templateGray[y_center_template:, :x_center_template]

            if region == NodeOrientation.SOUTH_EAST:
                region_image[y_center_region:, x_center_region:] = imageGray[y_center_region:, x_center_region:]
                region_template[y_center_template:, x_center_template:] = templateGray[y_center_template:, x_center_template:]

            # use ORB to detect keypoints and extract (binary) local
            # invariant features
            orb = cv2.ORB_create(maxFeatures)
            (kpsA_region, descsA) = orb.detectAndCompute(region_image, None)
            (kpsB_region, descsB) = orb.detectAndCompute(region_template, None)
            # match the features
            method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            matcher = cv2.DescriptorMatcher_create(method)
            matches_region = matcher.match(descsA, descsB, None)

            # add image patch offset for each match
            for match in matches_region:
                match.trainIdx += len(kpsA)
                match.queryIdx += len(kpsB)
                        
            # sort the matches by their distance (the smaller the distance,
            # the "more similar" the features are)
            matches_region = sorted(matches_region, key=lambda x:x.distance)

            # keep only the top matches
            #keep = int(len(matches_region) * keepPercent)
            #matches_region = matches_region[:4]

            kpsA.extend(kpsA_region)
            kpsB.extend(kpsB_region)
            matches.extend(matches_region)


        # divide image into x sub patch and find best match for each sub patch
        y_step, x_step = np.array(region_image.shape[:2]) // 8
        source_points = np.array([p.pt for p in kpsA])
        filtered_matches = []
        for x in range(0, region_image.shape[1], x_step):
            for y in range(0, region_image.shape[0], y_step):

                idxs = np.nonzero((source_points[:, 0] > y) & (source_points[:, 0] < y + y_step) & 
                                        (source_points[:, 1] > x) & (source_points[:, 1] < x + x_step))[0]

                #cv2.rectangle(source_image, (x,y), (x+x_step, y+y_step), (255,0,0), 3) 
                #cv2.rectangle(target_image, (x,y), (x+x_step, y+y_step), (255,0,0), 3) 

                if len(idxs) > 0:
                    for match in matches:
                        if match.queryIdx in idxs:
                            filtered_matches.append(match)
                            break
        matches = filtered_matches


        # check to see if we should visualize the matched keypoints
        matchedVis = None
        if debug:
            matchedVis = cv2.drawMatches(source_image, kpsA, target_image, kpsB, matches, None)
            
        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

            # scale points 
            for s_scale, t_scale in zip(source_scale, target_scale):
                ptsA[i] *= s_scale 
                ptsB[i] *= t_scale 

        return ptsA, ptsB, matchedVis

    def _get_detector_matcher(self, point_extractor="orb", maxFeatures:int=500, crossCheck:bool=False, flann:bool=False, **kwargs):
    
        if point_extractor == "orb":
            detector = cv2.ORB_create(maxFeatures)
            norm = cv2.NORM_HAMMING
        elif point_extractor == "sift":
            detector = cv2.SIFT_create() # maxFeatures
            norm = cv2.NORM_L2
        else:
            return None, None
        
        if flann:
            if norm == cv2.NORM_L2:
                flann_params = dict(algorithm = 1, trees = 5)
            else:
                flann_params= dict(algorithm = 6,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        else:
            matcher = cv2.BFMatcher(norm, crossCheck)
        return detector, matcher

    def _filter_matches(self, kp1, kp2, matches, ratio = 0.75, **kwargs):
        mkp1, mkp2, good = [], [], []
        for match in matches:
            if len(match) < 2:
                break
            
            m, n = match
            if m.distance < n.distance * ratio:
                good.append([m])
                mkp1.append(np.array(kp1[m.queryIdx].pt))
                mkp2.append(np.array(kp2[m.trainIdx].pt))

        return mkp1, mkp2, good 


    def extract_matching_points_old(self, source_image, target_image,  
                    debug=False, 
                    source_scale:[tuple]=[(1,1)], 
                    target_scale:[tuple]=[(1,1)],
                    point_extractor:callable="orb",
                    use_gray:bool=False, 
                    **kwargs
                    ):

        source_scale = np.array(source_scale)
        target_scale = np.array(target_scale)

        source_image = np.array(source_image) if type(source_image) == Image.Image else source_image
        target_image = np.array(target_image) if type(target_image) == Image.Image else target_image
        
        if callable(point_extractor):
            kpsA_ori, descsA, kpsB, descsB, matches = point_extractor(source_image, target_image)
        else:
            detector, matcher = self._get_detector_matcher(point_extractor=point_extractor, **kwargs)

            kpsA_ori, descsA = detector.detectAndCompute(source_image, None) if use_gray == False else detector.detectAndCompute(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), None)
            kpsB_ori, descsB = detector.detectAndCompute(target_image, None) if use_gray == False else detector.detectAndCompute(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), None)

            matches = matcher.knnMatch(descsA, descsB, k=2)
            kpsA, kpsB, matches = self._filter_matches(kpsA_ori, kpsB_ori, matches, **kwargs)

        # check to see if we should visualize the matched keypoints
        matchedVis = None
        if debug:
            #matchedVis = cv2.drawMatches(source_image, kpsA, target_image, kpsB, matches, None)
            matchedVis = cv2.drawMatchesKnn(source_image, kpsA_ori, target_image, kpsB_ori, matches, 
                                    None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
        ptsA, ptsB = [], []

        for ptA, ptB in zip(kpsA, kpsB):
            # scale points 
            for s_scale, t_scale in zip(source_scale, target_scale):
                ptA *= s_scale 
                ptB *= t_scale 

            ptsA.append(ptA)
            ptsB.append(ptB)

        return np.array(ptsA), np.array(ptsB), matchedVis

    def get_region_thumbnail(self, slide, boundary:Rect, size=(2048, 2048)):

        scale = []

        depth = self.depth + 1
        downsample = max(*[dim / thumb for dim, thumb in zip((boundary.w, boundary.h), (size[0] * depth, size[1] * depth))])        
        level = slide.get_best_level_for_downsample(downsample)

        downsample = slide.level_downsamples[level]

        x, y, w, h = int(boundary.west_edge), int(boundary.north_edge), int(boundary.w / downsample), int(boundary.h / downsample)
        scale.append(np.array((boundary.w, boundary.h)) / (w, h))

        tile = slide.read_region((x, y), level, (w, h))

        thumb = Image.new('RGB', tile.size, '#ffffff')
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, Image.ANTIALIAS)
        scale.append(np.array([w, h]) / thumb.size)

        return thumb, scale

    def __getstate__(self):

        attributes = self.__dict__.copy()

        del attributes['matchedVis']
        del attributes['_source_thumbnail']
        del attributes['_target_thumbnail']
        del attributes['tf_param']
        del attributes['_source_slide']
        del attributes['_target_slide']
        return attributes

    def __setstate__(self, state):

        self.__dict__ = state

        self._source_thumbnail = None
        self._target_thumbnail = None
        self._source_slide = None
        self._target_slide = None
        self.matchedVis = None

        self.tf_param = tf.AffineTransformation(self.__dict__["b"], self.__dict__["t"])

