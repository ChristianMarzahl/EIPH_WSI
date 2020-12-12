import numpy as np
from datetime import datetime
from sklearn.neighbors import KDTree

from Analysis.annotations import Annotation

from lib.BoundingBoxes import BoundingBoxes
from lib.utils import *

class Image:

    def __init__(self, file_name, user, bbType):

        self.user = user
        self.file_name = file_name
        self.annotations = []
        self.open_time_points = []
        self.close_time_points = []

        self.last_open = None
        self.last_close = None
        self.bbType = bbType

    @property
    def last_open_time_point(self):

        if self.last_open is None:
            open_time_points = sorted(self.open_time_points)

            if len(open_time_points) == 1:
                self.last_open = open_time_points[0]
            else:
                if len(self.annotations) > 0:
                    self.last_open = sorted([anno.Last_Edit_Time for anno in self.annotations])[-1]
                else:
                    self.last_open = None
            #for anno in self.annotations:
            #    if anno.Last_Editor == self.user: 
            #        for open_time_point in open_time_points:
            #            if open_time_point < anno.Last_Edit_Time: 
            #                self.last_open = open_time_point

        return self.last_open

    @property
    def last_close_time_point(self):

        if self.last_close is None:
            close_time_points = sorted(self.close_time_points)

            if len(close_time_points) == 1:
                self.last_close = close_time_points[0]
            else:
                if len(self.annotations) > 0:
                    self.last_close = sorted([anno.Last_Edit_Time for anno in self.annotations])[0]
                else:
                    self.last_close = None
            #for anno in self.annotations:
            #    if anno.Last_Editor == self.user: 
            #        for close_time_point in close_time_points:
            #            if close_time_point < anno.Last_Edit_Time:
            #                self.last_close = close_time_point

        return self.last_close

    @property
    def seconds_to_label(self):

        total_seconds = []

        last = None
        for anno in sorted([anno.Last_Edit_Time for anno in self.annotations]):
            if last is None:
                last = anno
            else:
                seconds = (anno - last).total_seconds()
                if seconds < 300:
                    total_seconds.append(seconds) 
                last = anno
            
        return sum(total_seconds)

    @property
    def FileName(self):
        return self.file_name

    @property
    def Labels(self):
        return [anno.Label for anno in self.annotations]

    @property
    def Annotations(self):
        return self.annotations

    @property
    def BB_Boxes(self):
        return [anno.BBox for anno in self.annotations]

    def add_line_information(self, line):

        if line.startswith("["):
            self.add_time_point(line)
        else:
            self.add_annotaion(line) 

    def add_annotaion(self, line):

        splits = line.split('|')
        if len(splits) > 5:
            anno = Annotation.create(line, self.bbType)
            if anno.BorderCell == False and anno.deleted == False:
                self.annotations.append(Annotation.create(line, self.bbType))

    def add_time_point(self, line):

        splits = line.replace('[','').replace(']','').split('|')

        user = splits[1]

        if user == self.user:
            datetime_str = splits[2].split('.')[0]
            datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

            operation = int(splits[3])

            if operation == 1: # Open
                self.add_open_time_point(datetime_object)
            else:
                self.add_close_time_point(datetime_object)


    def add_open_time_point(self, time_point):
        self.open_time_points.append(time_point)

    def add_close_time_point(self, time_point):
        self.close_time_points.append(time_point)


    def keep_annotations_with_num_votes(self, n:int=3, r:int=50):
        final_annotations = []

        temp_annotations = np.array(self.annotations)
        while (len(temp_annotations) > 0):

            query_anno = temp_annotations[0]

            centers = np.array([anno.Center for anno in temp_annotations])
            tree = KDTree(centers)         
            index_per_point = tree.query_radius([query_anno.Center], r=r)[0]

            # at least n votes
            if len(index_per_point) >= n:
                final_annotations.append(query_anno)

            temp_annotations = np.delete(temp_annotations, index_per_point, None)

        self.annotations = final_annotations




