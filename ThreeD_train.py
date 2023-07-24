import os
import numpy as np
import torch
import albumentations as A
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchmetrics

from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import MetricCollection, classification
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from torchvision import transforms
import matplotlib.pyplot as plt

MODEL_PATH = "3D_sampling_cnn.pth"
# os.environ['TORCH_HOME'] = '/app/.cache'


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

        # print("IMAGE:")
        # print(f'{self._main_path}/images/{img_name}')
        image = np.load(f'{self._main_path}/images/{img_name}')
        image = (image - image.min()) / (image.max() - image.min())

        # print("MASK:")
        # print(f'{self._main_path}/masks/{img_name}')
        mask = np.load(f'{self._main_path}/masks/{img_name}')
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # normalization 0-255

        # extract start and finish coordinates from the filename
        file_name = os.path.splitext(img_name)[0]
        filename_parts = file_name.split('_')
        start_x, start_y, start_z = int(filename_parts[3][2:]), int(filename_parts[4][2:]), int(filename_parts[5][2:])
        finish_x, finish_y, finish_z = int(filename_parts[6][2:]), int(filename_parts[7][2:]), int(filename_parts[8][2:])
        coords = [[start_x, start_y, start_z], [finish_x, finish_y, finish_z]]
        coords = transforms.ToTensor()(np.array(coords))

        transformed = self._transforms(image=image, mask=mask)
        # print(coords)
        # print(transformed['image'].float())
        # print(f'Image shape: {image.shape}')
        # print(f'Mask shape: {mask.shape}')

        # return image, mask, coords
        return transformed['image'].float(), transformed['mask'][None, ...].float(), coords.float()


class MapsDataModule(pl.LightningDataModule):
    def __init__(self, main_path: Path = Path('/home/czarek/mgr/3D_data/train'), batch_size: int = 1, test_size=0.15,
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


class ThreeD_UNet_cooler(pl.LightningModule):
    def __init__(self):
        super(ThreeD_UNet_cooler, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()

        self.base_model = models.video.r3d_18(pretrained=True)
        self.base_model.stem[0] = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.conv1 = self.base_model.stem
        self.conv2 = self.base_model.layer1
        self.conv3 = self.base_model.layer2
        self.conv4 = self.base_model.layer3
        self.conv5 = self.base_model.layer4

        # Coordinate layers
        self.conv_coords_512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_coords_256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_coords_128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_coords_64 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconv6 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv7 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv3d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv8 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x, coords):
        # Reshape input so it is (batch_size, channels, depth, height, width) = (batch_size, 1, 80, 80, 80)
        x = x.unsqueeze(1)

        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        print(coords.size())  # torch.Size([2, 1, 2, 3])
        coords_64 = self.conv_coords_64(coords)
        coords_x2 = F.interpolate(coords_64.unsqueeze(-1), size=x2.size()[2:], mode='trilinear', align_corners=True)
        x2 = torch.cat((x2, coords_x2), dim=1)
        coords_128 = self.conv_coords_128(coords_64)
        coords_x3 = F.interpolate(coords_128.unsqueeze(-1), size=x3.size()[2:], mode='trilinear', align_corners=True)
        x3 = torch.cat((x3, coords_x3), dim=1)
        coords_256 = self.conv_coords_256(coords_128)
        coords_x4 = F.interpolate(coords_256.unsqueeze(-1), size=x4.size()[2:], mode='trilinear', align_corners=True)
        x4 = torch.cat((x4, coords_x4), dim=1)
        coords_512 = self.conv_coords_512(coords_256)
        coords_x5 = F.interpolate(coords_512.unsqueeze(-1), size=x5.size()[2:], mode='trilinear', align_corners=True)
        x5 = torch.cat((x5, coords_x5), dim=1)

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
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        # self.log_dict(self.train_metrics(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, coords = batch
        y_hat = self(x, coords)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        # self.log_dict(self.val_metrics(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, coords = batch
        y_hat = self(x, coords)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        # self.log_dict(self.test_metrics(y_hat, y))

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
    base_path = Path('/home/czarek/mgr/3D_data/train')
    images_names = [image_path.name
                    for image_path in sorted((base_path / 'images').iterdir())]
    dataset = MapsDataset(base_path, images_names, transforms=transform)

    # test the dataset by printing some values
    idx = 0
    image, mask, coords = dataset[idx]
    print(f'Image shape: {image.shape}')
    print(f'Mask shape: {mask.shape}')
    print(f'Start coordinate: {coords[0][0]}')
    print(f'Finish coordinate: {coords[0][1]}')


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
    images, masks, coords = batch

    # inspect the shape and data type of the images and masks
    print(images.shape, images.dtype)
    print(masks.shape, masks.dtype)


def test_training():
    # instantiate your data module
    data_module = MapsDataModule()

    model = ThreeD_UNet_cooler()

    neptune = pl.loggers.neptune.NeptuneLogger(
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZDE5YmQyMy0xNzRmLTRlMTQtYTU3Yy0wMmVmOGQ5MmVjZjEifQ==',
        project='czarkoman/mgr-sampling-cnn'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{val_loss:.3f}',
        verbose=True
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True
    )

    trainer = pl.Trainer(
                         logger=neptune,
                         accelerator='gpu',
                         fast_dev_run=False,
                         log_every_n_steps=3,
                         devices=1,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         max_epochs=100)
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module, ckpt_path='best')
    neptune.run.stop()

    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    # test_dataset()
    # test_datamodule()
    test_training()
