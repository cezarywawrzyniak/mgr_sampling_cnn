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

with torch.no_grad():
    output = model(image, coords)
    # clipped = torch.clamp(output, min=0, max=1)
    # print(model)
# make_dot(output, params=dict(list(model.named_parameters()))).render("torchviz", format="png")

# model_graph = draw_graph(model, input_data=(image, coords), expand_nested=True, save_graph=True, filename='torchview')

visualized_image = np.array(image[0].detach().cpu().numpy())
# visualized_image = visualized_image.transpose((2, 1, 0))
image_indices = np.nonzero(visualized_image)

visualized_mask = np.array(mask[0, 0].detach().cpu().numpy())
visualized_mask = ((visualized_mask - visualized_mask.min()) / (visualized_mask.max() - visualized_mask.min())) * 255
# visualized_mask = visualized_mask.transpose((2, 1, 0))
mask_indices = np.nonzero(visualized_mask)
colors_mask = visualized_mask[mask_indices]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
ax.set_xlim(0, visualized_mask.shape[0])
ax.set_ylim(0, visualized_mask.shape[1])
ax.set_zlim(0, visualized_mask.shape[2])
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
ax.set_xlim(0, visualized_mask.shape[0])
ax.set_ylim(0, visualized_mask.shape[1])
ax.set_zlim(0, visualized_mask.shape[2])
plt.show()

visualized_output = np.array(output[0, 0].detach().cpu().numpy())
visualized_output = ((visualized_output - visualized_output.min()) / (visualized_output.max() - visualized_output.min())) * 255
# visualized_output = visualized_output.transpose((2, 1, 0))
threshold_output = 250
visualized_output_binary = (visualized_output > threshold_output)
visualized_output_masked = visualized_output.copy()
visualized_output_masked[~visualized_output_binary] = 0
print(visualized_output_masked.shape)

output_indices = np.nonzero(visualized_output_masked)
colors = visualized_output_masked[output_indices]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
ax.set_xlim(0, visualized_output_masked.shape[0])
ax.set_ylim(0, visualized_output_masked.shape[1])
ax.set_zlim(0, visualized_output_masked.shape[2])
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
ax.set_xlim(0, visualized_output_masked.shape[0])
ax.set_ylim(0, visualized_output_masked.shape[1])
ax.set_zlim(0, visualized_output_masked.shape[2])

plt.show()



