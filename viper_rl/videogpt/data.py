import os
import os.path as osp
import glob
from google.cloud import storage
import io
import cv2
import math
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from moviepy.editor import VideoFileClip
# import torch.distributed as dist

# dist.init_process_group(backend='nccl')

class VideoToImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.frame_indices = self._get_frame_indices()

    def _get_frame_indices(self):
        frame_indices = []
        for idx in range(len(self.dataset)):
            video = self.dataset[idx]['video']
            num_frames = video.shape[0]  # Assuming video shape is (frames, height, width, channels)
            for frame_idx in range(num_frames):
                frame_indices.append((idx, frame_idx))
        return frame_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        video_idx, frame_idx = self.frame_indices[idx]
        video = self.dataset[video_idx]['video']
        frame = np.transpose(video[frame_idx], (2,0,1))
        return {'image': frame}

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        video = self.dataset[idx]['video']
        axis_order = tuple(range(video.ndim - 3)) + (video.ndim - 1, video.ndim - 3, video.ndim - 2)
        video = np.transpose(video, axis_order)
        return {'video': video}
    
    def __len__(self):
        return len(self.dataset)
    

# class CombinedDataset(IterableDataset):
#     def __init__(self, datasets, dataset_labels, batch_size, batch_per_dset):
#         self.datasets = [iter(dataset) for dataset in datasets]  # Convert to iterators
#         self.dataset_labels = dataset_labels
#         self.batch_size = batch_size
#         self.batch_per_dset = batch_per_dset
#         self.N = len(datasets)  # Number of datasets

#     def __iter__(self):
#         idx = 0
#         while True:
#             batch = []
#             for _ in range(math.ceil(self.batch_size / self.batch_per_dset)):
#                 while self.datasets[idx] is None:
#                     idx = (idx + 1) % self.N
#                 x_i = next(self.datasets[idx])
#                 x_i['label'] = torch.full((self.batch_per_dset,), self.dataset_labels[idx], dtype=torch.int32)
#                 batch.append(x_i)
#                 idx = (idx + 1) % self.N

#             # Custom function to reshape the batch
#             yield self._reshape_batch(batch)

#     def _reshape_batch(self, batch):
#         # Assuming all items in the batch have the same structure
#         # Reshape the batch as per the original function's logic
#         batched_data = {}
#         for key in batch[0]:
#             data = [item[key] for item in batch]
#             concatenated_data = torch.cat(data, dim=0)[:self.batch_size]
#             batched_data[key] = concatenated_data.view(self.N, -1, *concatenated_data.shape[1:])
#         return batched_data

class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, paths, split):
        self.dataset = dataset
        self.split = split
        self.labels = self.assign_labels(paths)
        # print(len(self.labels))

    def assign_labels(self, paths):
        labels = []
        for idx, path in enumerate(paths):
            folder = os.path.join(path, self.split, '*.npz')
            fns = _glob_files(folder)
            # print(path)
            num_items = len(fns) # Assuming the dataset size can be inferred
            labels.extend([idx] * num_items)
        return labels

    def __getitem__(self, idx):
        data = self.dataset[idx]
        label = self.labels[idx]
        # print(label)
        data['label'] = label
        return data

    def __len__(self):
        return len(self.dataset)

def load_dataset(config, train, num_ds_shards, ds_shard_id, modality='video'):
    num_data_local = torch.cuda.device_count()
    # num_ds_shards = dist.get_world_size()
    # ds_shard_id = dist.get_rank()

    N = num_data_local
    batch_size = config.batch_size // num_ds_shards

    # initial_shape = (N, batch_size // N)
    initial_shape = None

    def get_dataset(data_path, data_type, batch_n, initial_shape, mask):
        load_fn = {
            'npz': load_npz,
            'mp4': load_mp4s,
        }[data_type]
        dataset = load_fn(
            config, data_path, train, num_ds_shards, ds_shard_id, mask
        )
        
        if modality == 'image':
            dataset = VideoToImageDataset(dataset)
        else:
            dataset = VideoDataset(dataset)
        # print(len(dataset))
        return dataset

    def get_data_type(data_path):
        fns = _glob_files(osp.join(data_path, '*'))
        fns = list(filter(lambda fn: not fn.endswith('pkl'), fns))
        if len(fns) == 2: # 'train' and 'test' directories
            fns = _glob_files(osp.join(data_path, 'test', '*'))
            if 'mp4' in fns[0]:
                return 'mp4', None
            elif 'npz' in fns[0]:
                return 'npz', None
            else:
                raise NotImplementedError(f'Unsupported file type {fns[0]}')
        else:
            return 'folder', fns

    data_type, aux = get_data_type(config.data_path)
    # print(aux)
    # print(data_type)
    if data_type in ['mp4', 'npz']:
        dataset = get_dataset(
            config.data_path, data_type, batch_size, initial_shape=initial_shape
        )
        class_map, mask_map = None, None

        data_loader = prepare(
            dataset, batch_size,
            initial_shape=initial_shape,
            num_device=N,
        )
    else:
        data_paths = aux
        class_map = {osp.basename(k): i for i, k in enumerate(data_paths)}

        mask_file = osp.join(config.data_path, 'mask_map.pkl')
        if os.path.exists(mask_file):
            with open(mask_file, 'rb') as f:
                mask_map = pickle.load(f)
        else:
            mask_map = None
        
        batch_per_dset = max(1, batch_size // len(data_paths))
        dataset_labels = list(range(len(data_paths)))
        if len(data_paths) >= num_ds_shards:
            data_paths = np.array_split(data_paths, num_ds_shards)[ds_shard_id].tolist()
            dataset_labels = np.array_split(dataset_labels, num_ds_shards)[ds_shard_id].tolist()

            # No need to shard further in load_* functions
            num_ds_shards = 1
            ds_shard_id = 0

        datasets = []
        for data_path in data_paths:
            data_type, _ = get_data_type(data_path)
            if mask_map is None:
                mask = None
            else:
                mask = mask_map[osp.basename(data_path)]
            d = get_dataset(data_path, data_type, batch_per_dset, initial_shape=None, mask=mask)
            print("Number of data is {}".format(len(d)))
            datasets.append(d)
        
        dataset = ConcatDataset(datasets)
        print("Total number of data is {}".format(len(dataset)))

        if modality == 'video':
            split = 'train' if train else 'test'
            dataset = LabelDataset(dataset, data_paths, split)


        data_loader = prepare(
            dataset, batch_size,
            initial_shape=initial_shape,
            num_device=N,
        )
        # dataset = ConcatDataset(datasets)
        # dataset = CombinedDataset(datasets, dataset_labels, batch_size, batch_per_dset)
        
    # dataset = jax_utils.prefetch_to_device(dataset, 2)
    return data_loader, class_map, mask_map


def prepare(dataset, batch_size, initial_shape=None, num_device=1):
    # Check if in DEBUG mode
    shuffle_size = batch_size if os.environ.get('DEBUG') == '1' else batch_size * 64

    # Create a data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffling the dataset
        drop_last=True, # Dropping the last incomplete batch
        num_workers=os.cpu_count() // num_device, # os.cpu_count(), # Utilizing multiple CPU cores
        prefetch_factor=None, # Prefetching batches
        pin_memory=True,
        collate_fn=lambda x: custom_collate_fn(x, initial_shape)
    )
    return data_loader


def custom_collate_fn(batch, initial_shape):
    # Reshape the data in the batch if initial_shape is specified
    # processed_batch = [_reshape_sample(sample, initial_shape) for sample in batch]
    return torch.utils.data.dataloader.default_collate(batch)

def _reshape_sample(sample, initial_shape):
    # Implement reshaping logic based on initial_shape
    # Assuming 'sample' is a dictionary with tensors
    reshaped_sample = {}
    for key, value in sample.items():
        # print(initial_shape)
        # print(value.shape)
        reshaped_sample[key] = value.view(*initial_shape, *value.shape[1:])
    return reshaped_sample


class NPZDataset(Dataset):
    def __init__(self, file_paths, config, mask=None):
        self.file_paths = file_paths
        self.config = config
        self.mask = mask

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load data from npz file
        # print(file_path)
        video = np.load(file_path)['arr_0']
        if self.mask is not None:
            video *= self.mask

        # Data processing
        if hasattr(self.config, 'seq_len'):
            req_len = 1 + (self.config.seq_len - 1) * self.config.frame_skip
            max_idx = video.shape[0] - req_len + 1
            max_idx = min(max_idx, req_len)
            idx = np.random.randint(0, max_idx)
            video = video[idx:idx+req_len:self.config.frame_skip]

        video = torch.tensor(video, dtype=torch.float32) / 127.5 - 1
        return {'video': video}

def load_npz(config, data_path, train, n_shards, shard_id, mask):
    split = 'train' if train else 'test'
    folder = os.path.join(data_path, split, '*.npz')
    fns = _glob_files(folder)
    random.Random(1234).shuffle(fns) 
    assert len(fns) > 0, f"Could not find any files for {folder}"
    fns = np.array_split(fns, n_shards)[shard_id].tolist()

    if mask is not None:
        mask = mask.astype(np.uint8)

    dataset = NPZDataset(fns, config, mask)
    # data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
    return dataset


class MP4Dataset(Dataset):
    def __init__(self, file_paths, config, mask=None):
        self.file_paths = file_paths
        self.config = config
        self.mask = mask

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video = self.read_video(self.file_paths[idx])
        video = self.process_video(video)
        return {'video': torch.from_numpy(video)}

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mask is not None:
                frame *= self.mask
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def process_video(self, video):
        if hasattr(self.config, 'seq_len'):
            req_len = 1 + (self.config.seq_len - 1) * self.config.frame_skip
            max_idx = min(video.shape[0] - req_len + 1, req_len)
            idx = random.randint(0, max_idx)
            video = video[idx:idx + req_len * self.config.frame_skip:self.config.frame_skip]
        else:
            video = video[None, ...]
        video = (video.astype(np.float32) / 127.5) - 1
        return video



def load_mp4s(config, data_path, train, n_shards, shard_id, mask):
    split = 'train' if train else 'test'
    folder = osp.join(data_path, split, '*.mp4')
    fns = _glob_files(folder)
    random.Random(1234).shuffle(fns) 
    random.shuffle(fns)
    assert len(fns) > 0, f"Could not find any files for {folder}"
    fns = np.array_split(fns, n_shards)[shard_id]

    if mask is not None:
        mask = mask.astype(np.uint8)

    dataset = MP4Dataset(fns, config, mask)
    # data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
    return dataset


def _glob_files(pattern):
    if pattern.startswith('gs://'):
        # Extract bucket name and blob pattern from the GCS pattern
        parts = pattern[5:].split('/', 1)
        bucket_name, blob_pattern = parts[0], parts[1]

        # Create a GCS client and get the bucket
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all blobs that match the pattern
        blobs = bucket.list_blobs(prefix=blob_pattern)
        fns = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
    else:
        # For local paths, use glob
        fns = list(glob.glob(pattern))

    fns.sort()
    return fns









