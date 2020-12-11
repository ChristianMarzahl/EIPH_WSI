import sqlite3
import numpy as np
from SlideRunner.dataAccess.database import Database
from tqdm import tqdm
from pathlib import Path
import openslide
import time
import pickle
import cv2
from glob import glob
import os

path = Path('/data/Datasets/EIPH_WSI/')

database = Database()
database.open(str(path/'EIPH.sqlite'))

size = 1024
level = 0

files = []

class SlideContainer():

    def __init__(self, file: Path, level: int=0, width: int=256, height: int=256):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = slide.level_downsamples[level]

        if level is None:
            level = slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
        return np.array(self.slide.read_region(location=(int(x * down_factor),int(y * down_factor)),
                                          level=self.level, size=(self.width, self.height)))[:, :, :3]

    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return str(self.path)


files = [fn for fn in glob(str(path/'*'/'*.svs'), recursive=True) if "11_" in fn or "20_" in fn or "22_" in fn]
container = []
getslides = """SELECT uid, filename FROM Slides"""
for filename in tqdm(files):

    check = True if 'erliner' in filename else False
    slidetype = 'Berliner Blau/' if check else 'Turnbull Blue/'

    slide_path = path / slidetype / filename

    slide = openslide.open_slide(str(slide_path))
    level = level#slide.level_count - 1
    level_dimension = slide.level_dimensions[level]
    down_factor = slide.level_downsamples[level]

    container.append(SlideContainer(slide_path, level, size, size))


folder = Path("/data/Datasets/EIPH_WSI/RCNN-Patches/Inference/")
for slide_container in tqdm(container):

    result_folder = folder/slide_container.file.stem

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for x in range(0, slide_container.slide.level_dimensions[level][1] - 2 * size, int(size / 2)):
        for y in range(0, slide_container.slide.level_dimensions[level][0] - 2 * size, int(size / 2)):
            patch_ori = slide_container.get_patch(x, y)

            patch_name = "{0}_{1}_{2}.png".format(x, y, size)
            cv2.imwrite(str(result_folder/patch_name), patch_ori[:, :, [2,1,0]])



