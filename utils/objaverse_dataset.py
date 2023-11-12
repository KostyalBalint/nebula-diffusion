import os
import random

import numpy as np
import torch
import trimesh as trimesh
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.objaverse_path import load_object_paths


class ObjaversePointCloudDataset(Dataset):
    def __init__(self, annotations_file, pc_dir, scale_mode, file_ext='.ply', load_to_mem=False, name_filter=None, transform=None):
        self.load_to_mem = load_to_mem
        self.pc_dir = pc_dir
        # { key(uid): path(base/[dir]/uid.ply) }
        self.object_paths = load_object_paths(helper_file_path='data', file_ext=file_ext)
        pc_dir_uids = set(self.read_uids_from_pc_folder(file_ext))

        # {[uid]: {'uid': str,
        #  'name': str,
        #  'description': str,
        #  'custom_label': str,
        #  'latent_text': [float] }}
        annotations = np.load(annotations_file, allow_pickle=True)

        # Filter out all the annotations where we don't have a point cloud file in the pc_dir folder
        # Filter out all the annotations by the name_filter
        filtered_annotations = {annotation['uid']: annotation for annotation in annotations if
                                annotation['uid'] in pc_dir_uids and (
                                            name_filter is None or name_filter in annotation['name'] or name_filter in annotation['custom_label'])}

        self.annotations = filtered_annotations

        self.uids = [d["uid"] for d in list(filtered_annotations.values()) if "uid" in d]

        if load_to_mem:
            self.pointclouds = self.load_all_pc_from_disk()

        # Deterministically shuffle the dataset
        self.uids.sort(reverse=False)
        random.Random(2023).shuffle(self.uids)

        self.scale_mode = scale_mode
        self.transform = transform

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        pc, uid = self.get_pc(idx)
        pc, shift, scale = self.scale_pc(pc)

        latent_text = self.annotations[uid]['latent_text']

        return {
            'pointcloud': pc,
            'latent_text': torch.tensor(latent_text, dtype=torch.float32),
            'id': uid,
            'shift': shift,
            'scale': scale
        }

    def get_pc(self, idx):
        if self.load_to_mem:
            # Load dataset from memory
            return self.pointclouds[idx]
        else:
            return self.load_pc_from_disk(idx)

    def get_file_path(self, idx):
        uid = self.uids[idx]
        return os.path.join(self.pc_dir, self.object_paths[uid])

    def load_pc_from_disk(self, idx):
        uid = self.uids[idx]
        path = os.path.join(self.pc_dir, self.object_paths[uid])
        if path.endswith('.ply'):
            pc = torch.tensor(trimesh.load(path).vertices, dtype=torch.float32)
        else:
            pc = torch.tensor(np.load(path)['arr_0'], dtype=torch.float32)
        return pc, uid

    def load_all_pc_from_disk(self):
        pcs = []
        print('Loading dataset into memory...')
        for i in tqdm(range(len(self.uids))):
            pcs.append(self.load_pc_from_disk(i))
        return pcs

    def scale_pc(self, pc):
        if self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        return (pc - shift) / scale, shift, scale

    def read_uids_from_pc_folder(self, file_ext):
        uids = []

        # Walk through the folder structure
        for root, dirs, files in os.walk(self.pc_dir):
            for file in files:
                if file.endswith(file_ext):
                    # Construct the full file path
                    file_path = os.path.join(root, file)

                    # Extract the 'uid' from the file name
                    uid = os.path.splitext(file)[0]

                    # Append the 'uid' to the list
                    uids.append(uid)

        return uids
