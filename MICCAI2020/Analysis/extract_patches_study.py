from glob import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import openslide
import time
import pandas as pd
from random import randint
import openslide
import cv2
from sklearn.neighbors import KDTree

path = Path(".")

patches_path = Path('D:/ProgProjekte/Python/Results-Exact-Study/Patches')

folder = Path('D:\\ProgProjekte\\Python\\Exact\\Converter\\result')
image_paths = glob(str(folder) + "\\mitosen\\*.tiff")
image_paths = [Path(path) for path in image_paths]

path_dict = {path.name:path for path in image_paths}

annotations = pd.read_pickle(str(path/"Results/StudyAnnotations.pkl"))
annotations = annotations[annotations['DatasetType']=='MitoticFigure']

for fileName in tqdm(path_dict.keys()):

    slide = openslide.open_slide(str(path_dict[fileName]))

    for mode in ['ExpertAlgorithm', 'Annotation', 'GroundTruth']:

        file_annos = annotations[(annotations['ProjectType'] == mode) & (annotations['FileName'] == fileName)]

        centers = []
        results = {}
        for _, name, vector, label, _, _ in file_annos.values.tolist():
            center = (vector[0] + (vector[2] - vector[0]) / 2, vector[1] + (vector[3] - vector[1]) / 2)

            if len(centers) > 0:
                tree = KDTree(centers)
                index_per_point = tree.query_radius([center], r=25)[0]

                if len(index_per_point) == 0:
                    results["{}-{}".format(int(center[0]), int(center[1]))] = {'Vector': vector, 'Label': label, 'Users': [name]}
                    centers.append(center)
                else:
                    center = centers[index_per_point[0]]
                    results["{}-{}".format(int(center[0]), int(center[1]))]['Users'].append(name)
            else:
                results["{}-{}".format(int(center[0]), int(center[1]))] = {'Vector': vector, 'Label': label, 'Users': [name]}
                centers.append(center)

        for anno_id in results:
            anno = results[anno_id]
            vector = anno['Vector']
            label = anno['Label']
            users = anno['Users']

            x = vector[0]
            y = vector[1]
            width = vector[2] - vector[0]
            height = vector[3] - vector[1]
            patch = np.array(slide.read_region(location=(int(x),int(y)),
                                          level=0, size=(width, height)))[:, :, :3]

            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            patch_folder = Path(patches_path/mode/"train"/label)
            patch_folder.mkdir(parents=True, exist_ok=True)

            #cv2.imwrite(str(patch_folder/"{}_{}.png".format(len(users), anno_id)), patch)