import os, datetime, numpy as np, pandas as pd
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, seed_everything
from sevir_dataloader import SEVIRDataLoader
from typing import *
from einops import rearrange
from sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS


ACTIVATIONS = {
    "ReLU": torch.nn.ReLU,
    "SiLU": torch.nn.SiLU,
    "Tanh": torch.nn.Tanh,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    # Add other activation functions as needed
}


class TransformsFixRotation(nn.Module):
    r"""
    Rotate by one of the given angles.

    Example: `rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])`
    """

    def __init__(self, angles):
        super(TransformsFixRotation, self).__init__()
        if not isinstance(angles, Sequence):
            angles = [angles, ]
        self.angles = angles

    def forward(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"


class SEVIRTorchDataset(TorchDataset):
    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "THWC",
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        # print('data', data.shape)
        
        # print('self.ret_contigous', self.ret_contiguous)
        # assert 1==0
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data

    def __len__(self):
        return self.sevir_dataloader.__len__()


class SEVIRLightningDataModule(LightningDataModule):
    def __init__(self,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "NTHWC",
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule_only
                 dataset_name: str = "sevir",
                 sevir_dir: str = None,
                 start_date: Tuple[int] = None,
                 train_test_split_date: Tuple[int] = (2019, 6, 1),
                 end_date: Tuple[int] = None,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 ):
        super(SEVIRLightningDataModule, self).__init__()
        
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        print('batch_size', batch_size)
        # assert 1==0
        self.num_workers = num_workers
        self.seed = seed
        if sevir_dir is not None:
            sevir_dir = os.path.abspath(sevir_dir)
        if dataset_name == "sevir":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevir_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevirlr":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            print('sevir_dir', sevir_dir)
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevirlr'.")
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir
        print(' self.sevir_dir',  self.sevir_dir)
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        self.start_date = datetime.datetime(*start_date) \
            if start_date is not None else None
        self.train_test_split_date = datetime.datetime(*train_test_split_date) \
            if train_test_split_date is not None else None
        self.end_date = datetime.datetime(*end_date) \
            if end_date is not None else None
        self.val_ratio = val_ratio

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            sevir_train_val = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=True,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
                ret_contiguous=self.ret_contiguous,)
        
            # Iterate through the dataset
            self.sevir_train, self.sevir_val = random_split(
                dataset=sevir_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed))
        if stage in (None, "test"):
            self.sevir_test = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=False,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode="0",
                ret_contiguous=self.ret_contiguous,)

    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.sevir_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.sevir_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)


def save_video(forecast, truth, oberrstdev, obs_p, obs_fn, level=0):
    # Coerce inputs to CPU float tensors first
    forecast_t = torch.as_tensor(forecast, dtype=torch.float32, device='cpu')   # (time, ens, 2, lat, lon)
    truth_t = torch.as_tensor(truth, dtype=torch.float32, device='cpu')   # (time, 2,  lat, lon)

    # Select level
    forecast_t = forecast_t[:, :, level, :, :]   # (time, ens, lat, lon)
    truth_t = truth_t[:, level, :, :]         # (time, lat, lon)

    # Observations (CPU tensor -> NumPy)
    obs = obs_fn(truth_t).numpy() + np.random.randn(*truth_t.shape) * oberrstdev  # (time, lat, lon)

    # RMSEs on CPU tensors, then to NumPy for formatting
    rmse_t = ((forecast_t - truth_t.unsqueeze(1)).pow(2).mean(dim=(2, 3)).sqrt())                # (time, ens)
    rmse_mean_t = ((forecast_t.mean(dim=1) - truth_t).pow(2).mean(dim=(1, 2)).sqrt())            # (time,)
    rmse = rmse_t.numpy()
    rmse_mean = rmse_mean_t.numpy()

    # For plotting, use NumPy arrays
    forecast_np = forecast_t.numpy()
    truth_np = truth_t.numpy()

    ens_mean = forecast_np.mean(axis=1)  # (time, lat, lon)
    ens_std  = forecast_np.std(axis=1)   # (time, lat, lon)
    members  = forecast_np[:, :4]        # (time, 4,   lat, lon)

    vmin = np.min(truth_np); vmax = np.max(truth_np)
    vmin_obs = np.min(obs);  vmax_obs = np.max(obs)

    fig, axs = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    cmap, norm, vmin, vmax = get_cmap("vil", encoded=True)

    img = []
    titles = ['Truth', 'Obs', 'Mean', 'Std', 'Member 1', 'Member 2', 'Member 3', 'Member 4']

    img.append(axs[0, 0].imshow(truth_np[0]*255, cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 1].imshow(obs[0]*255, cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 2].imshow(ens_mean[0]*255, cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 3].imshow(ens_std[0]*255, cmap=cmap, vmin=vmin, vmax=vmax))

    for i in range(4):
        img.append(axs[1, i].imshow(members[0, i]*255, cmap=cmap, vmin=vmin, vmax=vmax))

    for i, ax in enumerate(axs.flatten()):
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')

    sutitle = f't = {0}'
    fig.suptitle(sutitle, fontsize=14)

    def update(frame):
        img[0].set_array(truth_np[frame]*255)
        img[1].set_array(obs[frame]*255)
        img[2].set_array(ens_mean[frame]*255)
        img[3].set_array(ens_std[frame]*255)
        for i in range(4):
            img[4 + i].set_array(members[frame, i]*255)

        fig.suptitle(f't={frame}', fontsize=14, y=1.1)

        titles = [
            'Truth',
            f'Obs, $\\sigma$={np.round(oberrstdev, 1)}, p={np.round(obs_p*100, 1)}%',
            f'Mean ({rmse_mean[frame]:.2f})',
            'Std',
        ]
        for i in range(4):
            titles.append(f'Member {i+1} ({rmse[frame, i]:.2f})')

        for i, ax in enumerate(axs.flatten()):
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')

        return img

    fname = 'animation.mp4'
    ani = animation.FuncAnimation(fig, update, frames=truth_np.shape[0], interval=10, blit=True)
    ani.save(fname, writer='ffmpeg', fps=10, dpi=300)
    plt.close(fig)
    return fname
