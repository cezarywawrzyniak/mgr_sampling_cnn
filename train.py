import os
import numpy as np
import torch
import albumentations as A
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import MetricCollection, classification
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from pytorch_lightning import Trainer
from torchvision import transforms


class MapsDataset(Dataset):
    def __init__(self, main_path: Path, img_names: List[str], transforms: A.Compose):
        super().__init__()

        self._main_path = main_path
        self._img_names = img_names
        self._transforms = transforms

    def __len__(self):
        return len(self._img_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name = self._img_names[index]

        read_image = Image.open(self._main_path / 'images' / img_name)
        read_image = read_image.convert('RGB')
        image = np.asarray(read_image)
        # print("IMAGE:")
        # print(self._main_path / 'images' / img_name)

        read_mask = Image.open(self._main_path / 'masks' / img_name)
        mask = np.asarray(read_mask)
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # normalization 0-255
        # print("MASK:")
        # print(self._main_path / 'masks' / img_name)

        # extract start and finish coordinates from the filename
        file_name = os.path.splitext(img_name)[0]
        filename_parts = file_name.split('_')
        start_x, start_y = int(filename_parts[3][2:]), int(filename_parts[4][2:])
        finish_x, finish_y = int(filename_parts[5][2:]), int(filename_parts[6][2:])
        coords = [[start_x, start_y], [finish_x, finish_y]]
        coords = transforms.ToTensor()(np.array(coords))

        transformed = self._transforms(image=image, mask=mask)
        # print(coords)
        # print(transformed['image'].float())
        # print(f'Image shape: {image.shape}')
        # print(f'Mask shape: {mask.shape}')

        # return image, mask, coords
        return transformed['image'].float(), transformed['mask'][None, ...].float(), coords.float()


class MapsDataModule(pl.LightningDataModule):
    def __init__(self, main_path: Path = Path('/home/czarek/mgr/data/train'), batch_size: int = 3, test_size=0.15,
                 num_workers=16):
        super().__init__()
        self._main_path = main_path
        self._batch_size = batch_size
        self._test_size = test_size  # percentage
        self._num_workers = num_workers

        self.augmentations = A.Compose([
            ToTensorV2()
        ])
        self.transforms = A.Compose([
            ToTensorV2()
        ])

    def setup(self, stage: str):
        images_names = [image_path.name
                        for image_path in sorted((self._main_path / 'images').iterdir())]

        train_images_names, val_images_names = train_test_split(images_names,
                                                                test_size=self._test_size,
                                                                random_state=42)

        self.train_dataset = MapsDataset(self._main_path, train_images_names, self.augmentations)
        self.val_dataset = MapsDataset(self._main_path, val_images_names, self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size)


class UNet_cooler(pl.LightningModule):
    def __init__(self):
        super(UNet_cooler, self).__init__()

        self.base_model = models.resnet18(pretrained=True)

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = self.base_model.layer1
        self.conv3 = self.base_model.layer2
        self.conv4 = self.base_model.layer3
        self.conv5 = self.base_model.layer4

        # Decoder
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, coords):
        print(x.size())
        print(coords.size())

        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # x5 = torch.cat((x5, coords), dim=1)

        # Decoder
        x6 = self.upconv6(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.conv6(x6)
        x7 = self.upconv7(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.conv7(x7)
        x8 = self.upconv8(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.conv8(x8)
        x9 = self.final_conv(x8)
        return x9

    def training_step(self, batch, batch_idx):
        x, y, coords = batch
        y_hat = self(x, coords)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, coords = batch
        y_hat = self(x, coords)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, coords = batch
        y_hat = self(x, coords)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def test_dataset():
    # set up transforms
    transform = A.Compose([
        # transforms.ToPILImage(),
        ToTensorV2()
    ])

    # create dataset instance
    base_path = Path('/home/czarek/mgr/data/train')
    images_names = [image_path.name
                    for image_path in sorted((base_path / 'images').iterdir())]
    dataset = MapsDataset(base_path, images_names, transforms=transform)

    # test the dataset by printing some values
    idx = 0
    # image, mask, coords = dataset[idx]
    image, mask = dataset[idx]
    print(f'Image shape: {image.shape}')
    print(f'Mask shape: {mask.shape}')
    # print(f'Start coordinate: {coords["start"]}')
    # print(f'Finish coordinate: {coords["finish"]}')


def test_datamodule():
    # instantiate your data module
    data_module = MapsDataModule()
    data_module.setup("test")

    # load the training and validation sets using the datamodule's .train_dataloader() and .val_dataloader() methods
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # get a batch of training data
    batch = next(iter(train_dataloader))

    # extract the images and masks from the batch
    # images, masks, coordinates = batch
    images, masks = batch

    # inspect the shape and data type of the images and masks
    print(images.shape, images.dtype)
    print(masks.shape, masks.dtype)


def test_training():
    # instantiate your data module
    data_module = MapsDataModule()

    model = UNet_cooler()

    # neptune = pl.loggers.neptune.NeptuneLogger(
    #     api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZDE5YmQyMy0xNzRmLTRlMTQtYTU3Yy0wMmVmOGQ5MmVjZjEifQ==',
    #     project='czarkoman/zpo-project'
    # )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{val_loss:.3f}',
        verbose=True
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True
    )

    trainer = pl.Trainer(
                         # logger=neptune,
                         accelerator='gpu',
                         fast_dev_run=True,
                         # log_every_n_steps=3,
                         devices=1,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         max_epochs=1000)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    # test_dataset()
    # test_datamodule()
    test_training()