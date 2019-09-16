import numpy as np
import openslide
import cv2
import pickle
from glob import glob
from pathlib import Path
from tqdm import tqdm
import json

from sklearn.neighbors import KDTree


from Detection.data_loader import *

def non_max_suppression_by_distance(boxes, scores, radius: float = 25, return_ids = False):

    center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
    center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        id = sorted_ids[0]
        ids_to_keep.append(id)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[id]).nonzero()[0])

    return boxes[ids_to_keep] if return_ids == False else ids_to_keep



path = Path('/data/Datasets/EIPH_WSI/')
file_names =  [Path(path) for path in  glob(str(path/'*'/'*.svs'), recursive=True)]

turn = "Turnbull Blue"
ber = "Berliner Blau"

train = {
    "01": ber,
    "02": ber,
    "03": ber,
    "04": ber,
    "05": ber,
    "07": turn,
    "08": turn,
    "09": turn,
    "14": turn,
    "26": ber,
    "27": ber,
    "28": turn,
    "29": turn,
    "31": ber,
    "11": ber,
    "20": ber,
    "22": turn
}

annotations_container = pickle.load(open("inference_results_61.p", "rb"))


patch_size = 256
nr_cells = 15
level = 1


def get_patch(slide, x: int = 0, y: int = 0, width = 256, height = 256, down_factor = 1, level = 1):
    return np.array(slide.read_region(location=(int(x * down_factor), int(y * down_factor)),
                                           level=level, size=(width, height)))[:, :, :3]

def get_ids(annotations, xmin, ymin, xmax, ymax, patch_size):

    offset = patch_size / 2

    x = int(xmin - offset)
    y = int(ymin - offset)

    # select_boxes
    select_boxes = np.copy(annotations)
    select_boxes[:, [0, 2]] = select_boxes[:, [0, 2]] - x
    select_boxes[:, [1, 3]] = select_boxes[:, [1, 3]] - y

    bb_widths = (select_boxes[:, 2] - select_boxes[:, 0]) / 2
    bb_heights = (select_boxes[:, 3] - select_boxes[:, 1]) / 2

    ids = ((select_boxes[:, 0] + bb_widths) > 0) \
          & ((select_boxes[:, 1] + bb_heights) > 0) \
          & ((select_boxes[:, 2] - bb_widths) < patch_size) \
          & ((select_boxes[:, 3] - bb_heights) < patch_size)

    return  ids

color_lookup = {0:  [255, 255, 0, 255], 1:  [255, 0, 255, 255],
                2:  [0, 127, 0, 255], 3: [255, 127, 0, 255], 4:  [127, 127, 0, 255]}

offset = patch_size / 2



CATEGORIES = []
for i in range(5):
    CATEGORIES += {
        'id': str(i),
        'name': str(i),
        'supercategory': 'Hemosiderophages',
    },

coco_output = {
    # "info": INFO,
    #  "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

image_id = 1
annotation_id = 1

white_list = ['01_EIPH_563479 Turnbull blue', '02_EIPH_574162 Turnbull blue-001',
              '04_EIPH_567017 Turnbull blue-001', '08_EIPH_574999 Berliner Blau',
              '13_EIPH_570370 Berliner Blau', '14_EIPH_568381 berliner blau-001',
              '15_EIPH_568320 berliner blau', '15_EIPH_568320 Turnbull blue',
              '17_EIPH_575796 Turnbull blue', '19_EIPH_566933 L Berliner Blau',
              '19_EIPH_566933 L Tunrbull blue', '23_EIPH_563476 Berliner Blau-001',
              '23_EIPH_563476 Turnbull blue', '24_EIPH_576255 Turnbull blue',
              '25_EIPH_568150 Berliner Blau', '25_EIPH_568150 Turnbull blue',
              '27_EIPH_571557 Turnbull blue', '28_EIPH_569948 L berliner blau',
              '30_EIPH_568355 Turnbull blue', '30_EIPH_588355 Berliner Blau']

file_names = [fn for fn in file_names if fn.stem not in white_list]
for file_name in tqdm(file_names):
    id = file_name.name[:2]
    if id in train and \
        train[id] in str(file_name):
        continue


    slide = openslide.open_slide(str(file_name))

    level_dimension = slide.level_dimensions[level]
    down_factor = slide.level_downsamples[level]

    annotations = annotations_container[str(file_name)]

    annotations = non_max_suppression_by_distance(annotations, annotations[:, 4], radius=50)

    annotations[:, [0,1,2,3]] /= down_factor

    result_dict = {}
    found_patch = False

    for i in [4, 3, 2, 1, 0]:
        if found_patch:
            break
        for row in annotations[annotations[:, 4] == i]:
            x_min, y_min, x_max, y_max, c, conf = row

            ids = get_ids(annotations, x_min, y_min, x_max, y_max, patch_size)

            if np.count_nonzero(annotations[ids, 4] == 4) < 2:
                result_dict[np.count_nonzero(ids)] = row

        for count in [15, 16, 14, 13, 17, 18, 12, 11, 19, 20, 10]:
            if count in result_dict:
                x_min, y_min, x_max, y_max, c, conf = result_dict[count]

                ids = get_ids(annotations, x_min, y_min, x_max, y_max, patch_size)
                patch_annos = annotations[ids]
                patch_annos[:, [0, 2]] -= x_min
                patch_annos[:, [1, 3]] -= y_min

                patch_annos[:, [0, 2]] += offset
                patch_annos[:, [1, 3]] += offset


                patch = get_patch(slide, x_min - offset, y_min - offset, width=patch_size, height=patch_size,
                                  down_factor=down_factor, level=level)
                patch = patch.astype(np.uint8) #[:, :, [2, 1, 0]]

                image_info = {
                    "id": image_id,
                    "file_name": file_name.stem + ".png",
                    "width": patch_size,
                    "height": patch_size,
                }

                coco_output["images"].append(image_info)


                for rect in patch_annos:
                    x_min, y_min, x_max, y_max, c, conf = rect

                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": str(int(c)),
                        "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    }
                    coco_output["annotations"].append(annotation_info)
                    annotation_id = annotation_id + 1

                    #cv2.rectangle(patch, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                    #             color_lookup[int(c)], 2)



                image_id += 1
                cv2.imwrite("Bilder/" + file_name.stem + ".png", patch[:,:, [2,1,0]])
                found_patch = True
                break

with open(str('Bilder/WIPH_WSI_Coco.json'), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)


