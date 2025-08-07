# Sampling search space learning for mobile robot motion planning

This project was written during development of my master thesis at Poznań University of Technology. 

### Thesis summary:
This master's thesis explores the application of convolutional neural networks in the context of motion planning tasks. The main goal was to create a system that combines the advantages of a fully neural approach with those of traditional search space sampling algorithms. The neural network developed for this purpose generates a probability map of path occurrences in a given area, which is then used to guide the RRT* algorithm for sampling in promising regions of the map. This integration aimed to reduce the time required for path finding while maintaining the asymptotic optimality and probabilistic completeness inherent in the sampling method. The work involved developing a data generation scheme, designing the network architecture, and integrating these two main components. The foundational concept was initially devised for two-dimensional problems, but to demonstrate its potential in practical applications, the system was also extended to address three-dimensional problems.


Code is separated for 2D network and 3D network. Both cases have training code along with scripts to create training data step by step. 

### Creating 2D data step by step
![2D_data](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/refs/heads/main/architectures/english/data.svg)

### Operation of the 2D system
![2D_system](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/refs/heads/main/architectures/english/path_planning.svg)

### Path planning alghorithm
![neural_rrt](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/refs/heads/main/architectures/english/neural_rrt.svg)

### 2D neural architecture
![2D_architecture](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/refs/heads/main/architectures/english/2D_architecture.svg)

### Entire 3D system
![3D_system](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/90d34ca67bb9b436a22190513e6046b68d60e6b2/architectures/english/3D_system.svg)

### 3D nerual architecture
![3D_architecture](https://raw.githubusercontent.com/cezarywawrzyniak/mgr_sampling_cnn/90d34ca67bb9b436a22190513e6046b68d60e6b2/architectures/english/3D_architecture.svg)

## Results 
![results_1](https://github.com/czarkoman/mgr_sampling_cnn/blob/main/architectures/english/Slide15.jpg?raw=true)
![results_2](https://github.com/czarkoman/mgr_sampling_cnn/blob/main/architectures/english/Slide16.jpg?raw=true)
![results_3](https://github.com/czarkoman/mgr_sampling_cnn/blob/main/architectures/english/Slide17.jpg?raw=true)
