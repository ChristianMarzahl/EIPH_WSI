import json
import numpy as np
from datetime import datetime
from enum import Enum

from lib.BoundingBox import BoundingBox
from lib.utils import *



class Annotation:

    def __init__(self, image_name, label, x_min, y_min, x_max, y_max, first_editor, creation_time, last_editor, last_edit_time, uuid, deleted, anno_type :BBType, image_width: int, image_height:int):

        self.offset = 2
        self.is_border_cell = None

        self.image_name = image_name
        self.label = label
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.first_editor = first_editor
        self.creation_time = creation_time
        
        self.last_editor = last_editor
        self.last_edit_time = last_edit_time

        self.uuid = uuid
        self.deleted = deleted
        self.anno_type = anno_type

        self.image_width = image_width
        self.image_height = image_height

    @classmethod
    def create(cls, line: str, anno_type :BBType):

        splits = line.split('|') 

        image_name = splits[0]
        label = splits[1]

        vector = json.loads(splits[2].replace(",}", '}'))
        x_min = int(vector['x1'])
        y_min = int(vector['y1'])
        x_max = int(vector['x2'])
        y_max = int(vector['y2'])

        first_editor = splits[3]
        creation_time = Annotation.convert_date(splits[4].split('.')[0])
        
        last_editor = splits[5]
        last_edit_time = Annotation.convert_date(splits[6].split('.')[0])

        uuid = splits[7]
        deleted = "True" in splits[8]

        image_width = int(splits[9])
        image_height  = int(splits[10])

        return cls(image_name, label, x_min, y_min, x_max, y_max, first_editor, creation_time, last_editor, last_edit_time, uuid, deleted, anno_type, image_width, image_height)

    @property
    def BorderCell(self):

        if self.is_border_cell is None:
            self.is_border_cell = True

            if self.x_min > self.offset and self.x_max < self.image_width - self.offset:
                if self.y_min > self.offset and self.y_max < self.image_height - self.offset:
                    self.is_border_cell = False

        return self.is_border_cell

    @property
    def Label(self):
        return self.label 

    @property
    def Vector(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    @property
    def Vector_with_label(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.Label])

    @property
    def UUID(self):
        return self.uuid 

    @property
    def Deleted(self):
        return self.deleted 

    @property
    def First_Editor(self):
        return self.first_editor 

    @property
    def Creation_Time(self):
        return self.creation_time 

    @property
    def Last_Editor(self):
        return self.last_editor 

    @property
    def Last_Edit_Time(self):
        return self.last_edit_time

    @property
    def Center(self):
        return (self.x_min + (self.x_max - self.x_min) / 2, self.y_min + (self.y_max - self.y_min) / 2)

    @property
    def BBox(self):

        return BoundingBox(imageName=self.image_name, classId=self.Label, x=self.x_min,
                                   y=self.y_min, w=self.x_max - self.x_min, h=self.y_max - self.y_min,
                                   typeCoordinates=CoordinatesType.Absolute, classConfidence=1,
                                   bbType=self.anno_type, format=BBFormat.XYWH)

    @staticmethod
    def convert_date(datetime_str: str):
        return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')