import random
from copy import copy
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from tqdm.auto import tqdm


class ShapeNetCoreOwn(Dataset):

    def __init__(self, path, annotations_path, split, scale_mode, args, transform=None,):
        super().__init__()

        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34', 'unit_sphere')
        self.path = path
        self.annotations_path = annotations_path

        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.args = args

        self.pointclouds = []

        self.load()

    def load(self):

        annotations = np.load(self.annotations_path, allow_pickle=True).item()  ## dict_keys(['taxonomy_map', 'tokenized_taxonomy'])
        print('Loaded annotations')

        ann_map = {item['id']: annotations['tokenized_taxonomy'][item['category']]['tokens'] for item in annotations['taxonomy_map']}

        def map_pc(pc_id, pc):
            pc, shift, scale = self.scale_pc(torch.tensor(pc.vertices, dtype=torch.float32).to(self.args.device))
            return {
                'pointcloud': pc,
                'latent_text': ann_map[pc_id],
                'id': pc_id,
                'shift': shift,
                'scale': scale
            }

        print('Loading pcs')
        point_clouds = np.load(self.path, allow_pickle=True)['arr_0'].item()
        print('Loaded pcs')
        self.pointclouds = [map_pc(id, pc) for id, pc in tqdm(point_clouds.items())]


        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2023).shuffle(self.pointclouds)

        generator = torch.Generator().manual_seed(42)
        split_data = random_split(self.pointclouds, [0.85, 0.15], generator=generator)

        if self.split is not None:
            if self.split == 'train':
                self.pointclouds = list(split_data[0])
            if self.split == 'val':
                self.pointclouds = list(split_data[1])

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
        elif self.scale_mode == 'unit_sphere':
            # Shift to the center of the point cloud
            shift = pc.mean(dim=0).reshape(1, 3)

            # Scale to fit into a unit sphere
            max_dist = torch.sqrt(((pc - shift) ** 2).sum(dim=1)).max()
            scale =  max_dist / torch.ones([1, 1]).to(self.args.device)
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        return (pc - shift) / scale, shift, scale

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

