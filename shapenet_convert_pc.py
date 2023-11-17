import json
import os
from os import path
from contextlib import closing
from tqdm.auto import tqdm

import numpy as np
import trimesh

from multiprocessing import Pool

taxonomy = json.load(open('data/shapenetcore.taxonomy.json'))
tax_map = { tax['metadata']['name']:tax['metadata']['label'] + '\n' + tax['li_attr']['title'].replace('\n', '').strip() for tax in taxonomy}
# Add missing key for cellphone
tax_map['02992529'] = "cellphone,mobile,mobilephone,phone\na handheld device used by people for telecommunication"


def find_obj_files(folder_path):
    obj_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".obj"):
                obj_files.append(os.path.join(root, file))
    return obj_files

obj_files = find_obj_files('data/ShapeNetCore_unziped')

tax = [{
    'id': file.split('/')[4],
    'category': file.split('/')[2],
    'obj': '/'.join(file.split('/')[2:]),
    'text': tax_map[file.split('/')[3]]
} for file in obj_files]

def object_to_pointCloud(file, point_count=4096):
    mesh = trimesh.load(file_obj = open(file),
                        file_type='obj',
                        force='mesh',
                        skip_texture=False)

    samples, _ = trimesh.sample.sample_surface(mesh, point_count, sample_color=False)
    del mesh
    del _
    pc = trimesh.points.PointCloud(samples, colors=np.tile(np.array([0, 0, 0, 1]), (len(samples), 1)))
    del samples
    return pc

def process_function(obj):
    return object_to_pointCloud(path.join('data/ShapeNetCore_unziped', obj['obj']), point_count=4094 * 2), obj

if __name__ == "__main__":
    pcs = {}

    with closing(Pool(32)) as pool:
        with tqdm(total=len(tax)) as pbar:
            # Map tasks to worker_function and store results
            for result in pool.imap_unordered(process_function, tax):
                res, obj = result
                pcs[obj['id']] = res
                pbar.update(1)

    np.savez_compressed(f'data/shapenet_pc', pcs)
    print('Done')