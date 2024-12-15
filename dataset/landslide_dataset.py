import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py

class CustomLandslideDataset(data.Dataset):
    def __init__(self, data_directory, file_list_path, max_iterations=None, mode='label'):
        self.file_list_path = file_list_path
        self.data_mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.data_std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.mode = mode

        # Load image IDs from the file list
        self.image_ids = [line.strip() for line in open(file_list_path)]

        # Extend the image list if max_iterations is provided
        if max_iterations is not None:
            repeat_count = int(np.ceil(max_iterations / len(self.image_ids)))
            self.image_ids = self.image_ids * repeat_count + self.image_ids[:max_iterations - repeat_count * len(self.image_ids)]

        self.file_records = []

        if mode == 'labeled':
            for img_name in self.image_ids:
                img_path = f"{data_directory}{img_name}"
                label_path = img_path.replace('img', 'mask').replace('image', 'mask')
                self.file_records.append({'img': img_path, 'label': label_path, 'name': img_name})
        elif mode == 'unlabeled':
            for img_name in self.image_ids:
                img_path = f"{data_directory}{img_name}"
                self.file_records.append({'img': img_path, 'name': img_name})

    def __len__(self):
        return len(self.file_records)

    def __getitem__(self, index):
        record = self.file_records[index]

        # Load image data
        with h5py.File(record['img'], 'r') as h5_file:
            image = h5_file['img'][:]

        image = np.asarray(image, dtype=np.float32).transpose((-1, 0, 1))

        # Normalize the image data
        for channel_idx in range(len(self.data_mean)):
            image[channel_idx, :, :] = (image[channel_idx, :, :] - self.data_mean[channel_idx]) / self.data_std[channel_idx]

        if self.mode == 'labeled':
            # Load label data
            with h5py.File(record['label'], 'r') as h5_file:
                label = h5_file['mask'][:]

            label = np.asarray(label, dtype=np.float32)
            return image.copy(), label.copy(), np.array(image.shape), record['name']
        else:
            return image.copy(), np.array(image.shape), record['name']

if __name__ == '__main__':
    # Dataset and DataLoader setup
    dataset_directory = '/scratch/Land4Sense_Competition/'
    file_list = './train.txt'
    batch_size = 1

    landslide_dataset = CustomLandslideDataset(data_directory=dataset_directory, file_list_path=file_list, mode='labeled')
    data_loader = DataLoader(dataset=landslide_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Compute dataset statistics
    total_channels_sum = 0
    total_channel_squared_sum = 0
    num_batches = len(data_loader)

    for batch_data, _, _, _ in data_loader:
        total_channels_sum += torch.mean(batch_data, dim=[0, 2, 3])
        total_channel_squared_sum += torch.mean(batch_data ** 2, dim=[0, 2, 3])

    # Calculate mean and standard deviation
    calculated_mean = total_channels_sum / num_batches
    calculated_std = (total_channel_squared_sum / num_batches - calculated_mean ** 2) ** 0.5

    print("Mean:", calculated_mean.tolist())
    print("Std Dev:", calculated_std.tolist())
