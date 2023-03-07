# 🧊 3D Indoor Environment Reconstruction using Wi-Fi Sensing 📡

This work was produced as thesis for the master degree in Computer Science and aims at producing a 3D reconstruction of the environment only relying on the [Wi-Fi signal](https://en.wikipedia.org/wiki/WiFi_Sensing).

> At the moment of writing this is the first work which tries to achieve this task.

A novel dataset was collected in order to train a model and present a first solution for this task. The proposed method is based on a sanitization procedure on both amplitude and phase on the collected Wi-Fi data and a transformer-based architecture for the deep model. The results produced by the proposed architecture have been validated both from a qualitative and quantitative point of view, showing how the CSI data can actually carry enough information to discriminate and reconstruct environments.

## Architecture
The architecture proposed by this work to address 3D indoor environment reconstruction from Wi-Fi data is based on the original transformer architecture and it is composed by three modules: a 2D transformer encoder, a 3D transformer decoder and a 3D CNN upscaler.

![Architecture](https://user-images.githubusercontent.com/58000595/222985955-c786013b-8167-43bf-9f50-65d398d96574.png)

## Results
On the left are shown the predictions produced by the model and on the right the respective ground truth.

<img src="https://user-images.githubusercontent.com/58000595/222986049-12ad8b76-eaac-4065-8a5d-d2c916972ad9.png" alt="pred0" width="49%"></img>
<img src="https://user-images.githubusercontent.com/58000595/222986064-28ae85c6-2fbf-4657-9614-895baa4b9a7b.png" alt="true0" width="49%"></img>

<img src="https://user-images.githubusercontent.com/58000595/222986055-3be2f729-4863-43cd-85cc-dfeeb740960d.png" alt="pred1" width="49%"></img>
<img src="https://user-images.githubusercontent.com/58000595/222986066-fc2a9177-6d5c-409e-b77c-61d96bdc9e96.png" alt="true1" width="49%"></img>

In the following table, the results of the best model on train, validate and test set are reported.

| Set | Dice | Binary Cross-Entropy | Intersection over Union |
|:---:|:---:|:---:|:---:|
| Train | 0.500 | 0.002 | 0.996 |
| Validate | 0.514 | 0.023 | 0.958 |
| Test | 0.509 | 0.012 | 0.971 |

## Content

The thesis produced for this task is available in [`thesis.pdf`](thesis/thesis.pdf).

Regarding the implementation, in the `code` folder is available the file used to collect the data [`run_test.py`](code/run_test.py).

Moreover, in the `python` folder you can find the following content:
- the folder `dataset` contains the whole dataset used with both CSI data as input and 3D models as output;
- the folder `models` contains all the different models tested during the thesis;
- the folder `visualizer` contains the implementation of the Three.js visualizer used for the interactive visualization of the 3D models in a Jupyter notebook;
- the notebook `Tesi_Magistrale.ipynb` contains the training pipeline used in order to perform the experiments;
- the file `dataset.py`contains the necessary code to download and load the dataset;
- the file `plot.py` is used to produce several plots of the CSI;
- the file `preprocessing.py`contains the code to preprocess the CSI data;
- the file `utils.py` contains all the utility code.

Check the [PyTorch documentation](https://pytorch.org/get-started/locally/) in order to train the models with CUDA locally.

## Author

<a href="https://github.com/dotmat3" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Matteo%20Orsini-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>

## Technologies

In this project the following technologies were used:
- PyTorch for the implementation of the models
- PyTorchLightning for the training pipeline
- Weights and Biases for the tracking of the experiments
- Three.js for the interactive visualizer of the 3D models
