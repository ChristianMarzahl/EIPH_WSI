from glob import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import openslide
import time
import pandas as pd
from random import randint
import random
import openslide
import cv2
from SlideRunner.dataAccess.database import Database

patches_path = Path('D:/ProgProjekte/Python/Results-Exact-Study/Patches')
path = Path('./')

database = Database()
database.open(str('C:/Users/c.marzahl/Downloads/MITOS_WSI_CCMCT_MEL.sqlite'))

slide_names = ['f26e9fcef24609b988be.svs', 'f3741e764d39ccc4d114.svs', 'fff27b79894fe0157b08.svs']
slidelist_test = ['27', '30', '31', '6', '18', '20', '1', '2', '3' ,'9', '11']
nr_target_cells = 350

getslides = """SELECT uid, filename FROM Slides"""
for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
    if (str(filename) not in slide_names):
        continue

    database.loadIntoMemory(currslide)

    slide_path = Path('D:/Datasets/WSI-CCMCT') / filename
    slide = openslide.open_slide(str(slide_path))

    level = 0
    down_factor = 1

    classes = {2: "mitotic figure", 4: "non-mitotic cell"}

    nr_mitotic_figures = 0
    nr_nmc = 0

    keys = list(database.annotations.keys())
    random.shuffle(keys)
    for key in keys:
        annotation = database.annotations[key]

        if nr_nmc > nr_target_cells and nr_mitotic_figures > nr_target_cells:
            break

        if annotation.agreedClass in classes:
            label = classes[annotation.agreedClass]
            if label == "mitotic figure":
                nr_mitotic_figures += 1
                if nr_mitotic_figures > nr_target_cells:
                    continue
            else:
                nr_nmc += 1
                if nr_nmc > nr_target_cells:
                    continue

            annotation.r = 25
            d = int(2 * annotation.r / down_factor)
            x_min = int((annotation.x1 - annotation.r) / down_factor)
            y_min = int((annotation.y1 - annotation.r) / down_factor)

            patch = np.array(slide.read_region(location=(int(x_min),int(y_min)),
                                    level=0, size=(d, d)))[:, :, :3]
        
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            for mode in ['ExpertAlgorithm', 'Annotation', 'GroundTruth']:
                patch_folder = Path(patches_path/mode/"val"/label)
                patch_folder.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(patch_folder/"{}-{}.png".format(x_min, y_min)), patch)

            






