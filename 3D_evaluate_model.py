import torch

from pathlib import Path
from ThreeD_train import ThreeD_UNet_cooler, MapsDataModule
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchviz import make_dot
from torchview import draw_graph

MODEL_PATH = "3D_sampling_cnn.pth"
BASE_PATH = Path('/home/czarek/mgr/3D_eval_data')

model = ThreeD_UNet_cooler()
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()


data_module = MapsDataModule(main_path=BASE_PATH)
data_module.setup("test")

batch_size = 1
sampler = torch.utils.data.RandomSampler(data_module.train_dataset)

dataloader = torch.utils.data.DataLoader(
    data_module.train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=data_module._num_workers
)

batch = next(iter(dataloader))
# print(batch)
image, mask, coords = batch
print(coords.data.tolist()[0][0])
image_show = image.detach().cpu().numpy()
image_show = image_show.transpose((0, 2, 3, 1))
# print(image_show)
# plt.imshow(image_show[0])
# plt.show()

with torch.no_grad():
    output = model(image, coords)
    # clipped = torch.clamp(output, min=0, max=1)
    # print(model)
# make_dot(output, params=dict(list(model.named_parameters()))).render("torchviz", format="png")

# model_graph = draw_graph(model, input_data=(image, coords), expand_nested=True, save_graph=True, filename='torchview')

visualized_image = np.array(image[0].detach().cpu().numpy())
visualized_image = visualized_image.transpose((1, 2, 0))
image_indices = np.nonzero(visualized_image)

visualized_mask = np.array(mask[0, 0].detach().cpu().numpy())
visualized_mask = visualized_mask.transpose((1, 2, 0))
mask_indices = np.nonzero(visualized_mask)
colors_mask = visualized_image[mask_indices]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='r', marker='o')
ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')

# Set plot limits based on the image dimensions
ax.set_xlim(0, visualized_mask.shape[0])
ax.set_ylim(0, visualized_mask.shape[1])
ax.set_zlim(0, visualized_mask.shape[2])

plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
# Set plot limits based on the image dimensions
ax.set_xlim(0, visualized_mask.shape[0])
ax.set_ylim(0, visualized_mask.shape[1])
ax.set_zlim(0, visualized_mask.shape[2])
plt.show()

visualized_output = np.array(output[0, 0].detach().cpu().numpy())
visualized_output = visualized_output.transpose((1, 2, 0))
threshold_output = -3.0
visualized_output = (visualized_output > threshold_output).astype(np.uint8)

output_indices = np.nonzero(visualized_output)
colors = visualized_output[output_indices]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='r', marker='o')
ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')

# Set plot limits based on the image dimensions
ax.set_xlim(0, visualized_output.shape[0])
ax.set_ylim(0, visualized_output.shape[1])
ax.set_zlim(0, visualized_output.shape[2])

plt.show()



