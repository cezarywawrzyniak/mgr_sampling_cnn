import torch

from pathlib import Path
from ThreeD_train import ThreeD_UNet_cooler, MapsDataModule
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchviz import make_dot
from torchview import draw_graph

MODEL_PATH = "3D_sampling_cnn_vol2_47.pth"
BASE_PATH = Path('/home/czarek/mgr/3D_eval_data/')

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
start = coords.data.tolist()[0][0][0]
finish = coords.data.tolist()[0][0][1]
print(f'Start X:{start[0]}, Y:{start[1]}, Z:{start[2]}')
print(f'Finish X:{finish[0]}, Y:{finish[1]}, Z:{finish[2]}')

with torch.no_grad():
    output = model(image, coords)
    # print(model)
# make_dot(output, params=dict(list(model.named_parameters()))).render("torchviz", format="png")

# model_graph = draw_graph(model, input_data=(image, coords), expand_nested=True, save_graph=True,
#                          filename='torchview_3D')

visualized_image = np.array(image[0].detach().cpu().numpy())
visualized_image = visualized_image.transpose((1, 2, 0))
image_indices = np.nonzero(visualized_image)

voxels_mask = visualized_image.astype(bool)
voxels_color = np.zeros(visualized_image.shape + (4,))
voxels_color[voxels_mask] = (0, 0, 0, 0.2)

visualized_mask = np.array(mask[0, 0].detach().cpu().numpy())
visualized_mask = ((visualized_mask - visualized_mask.min()) / (visualized_mask.max() - visualized_mask.min())) * 255
mask_indices = np.nonzero(visualized_mask)
colors_mask = visualized_mask[mask_indices]

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
# ax.set_xlim(0, visualized_mask.shape[0])
# ax.set_ylim(0, visualized_mask.shape[1])
# ax.set_zlim(0, visualized_mask.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
# ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
# ax.set_xlim(0, visualized_mask.shape[0])
# ax.set_ylim(0, visualized_mask.shape[1])
# ax.set_zlim(0, visualized_mask.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

visualized_output = np.array(output[0, 0].detach().cpu().numpy())
visualized_output = ((visualized_output - visualized_output.min()) / (visualized_output.max() - visualized_output.min())) * 1.0
# visualized_output = visualized_output.transpose((1, 2, 0))
threshold_output = 0.96
visualized_output_binary = (visualized_output > threshold_output)
visualized_output_masked = visualized_output.copy()
visualized_output_masked[~visualized_output_binary] = 0
unique_values = list(set(visualized_output_masked.flatten()))
print(unique_values)

output_base_indices = np.nonzero(visualized_output)
colors_base = visualized_output[output_base_indices]
output_indices = np.nonzero(visualized_output_masked)
colors = visualized_output_masked[output_indices]

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
# ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
# ax.set_xlim(0, visualized_output_masked.shape[0])
# ax.set_ylim(0, visualized_output_masked.shape[1])
# ax.set_zlim(0, visualized_output_masked.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
# ax.set_xlim(0, visualized_output_masked.shape[0])
# ax.set_ylim(0, visualized_output_masked.shape[1])
# ax.set_zlim(0, visualized_output_masked.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# # Set of 4 plots
# fig = plt.figure(figsize=(20, 5))
#
# # Create four subplots with 3D projections
# ax1 = fig.add_subplot(141, projection='3d')
# ax2 = fig.add_subplot(142, projection='3d')
# ax3 = fig.add_subplot(143, projection='3d')
# ax4 = fig.add_subplot(144, projection='3d')
#
# # Scatter plots
# ax1.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
# ax2.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
# ax3.scatter(output_base_indices[0], output_base_indices[1], output_base_indices[2], c=colors_base, cmap='jet', marker='o')
# ax4.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
#
# # Set axis limits
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim(0, visualized_output_masked.shape[0])
#     ax.set_ylim(0, visualized_output_masked.shape[1])
#     ax.set_zlim(0, visualized_output_masked.shape[2])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
# plt.tight_layout()
# plt.show()


# Set of 3 plots
fig = plt.figure(figsize=(15, 5))

# Create four subplots with 3D projections
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Scatter plots
ax1.scatter(image_indices[0], image_indices[1], image_indices[2], c='k', marker='o')
# ax1.voxels(voxels_mask, facecolors=voxels_color, edgecolors=voxels_color)
ax2.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=colors_mask, cmap='jet', marker='o')
ax3.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')

# Set axis limits
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, visualized_output_masked.shape[0])
    ax.set_ylim(0, visualized_output_masked.shape[1])
    ax.set_zlim(0, visualized_output_masked.shape[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()



