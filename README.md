
# Sound event localization and detection of overlapping sources using convolutional recurrent neural network (SELDnet)
Sound event localization and detection (SELD) is the combined task of identifying the temporal onset and offset of a sound event, tracking the spatial location when active, and further associating a textual label describing the sound event.
The paper describing the SELDnet can be found [here](https://arxiv.org/abs/1807.00129 'Paper on arxiv'). We are releasing a simple vanila code without much frills and the related datasets here.
The work presented in this paper is an extension of the previous multichannel sound event detection, and direction of arrival estimation papers listed below.

1. Sound event detection (SED)
   - Sharath Adavanne, Giambattista Parascandolo, Pasi Pertila, Toni Heittola and Tuomas Virtanen, '[Sound event detection in multichannel audio using spatial and harmonic features](https://arxiv.org/abs/1706.02293 "Paper on arxiv.org")' at *Detection and Classification of Acoustic Scenes and Events (DCASE 2016)*
   - Sharath Adavanne, Pasi Pertila and Tuomas Virtanen, '[Sound event detection using spatial features and convolutional recurrent neural network](https://arxiv.org/abs/1706.02291  "Paper on arxiv.org")' at *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2017)* [[Code]](https://github.com/sharathadavanne/multichannel-sed-crnn "Code on Github")
   - Sharath Adavanne, Archontis Politis and Tuomas Virtanen, '[Multichannel sound event detection using 3D convolutional neural networks for learning inter-channel features](https://arxiv.org/abs/1801.09522  "Paper on arxiv.org")' at *International Joint Conference on Neural Networks (IJCNN 2018)*

2. Direction of arrival (DOA) estimation
   - Sharath Adavanne, Archontis Politis and Tuomas Virtanen, '[Direction of arrival estimation for multiple sound sources using convolutional recurrent neural network](https://arxiv.org/abs/1710.10059  "Paper on arxiv.org")' at *European Signal Processing Conference (EUSIPCO 2018)*

## More about SELDnet
The proposed SELDnet architecture is as shown below. The input is the multichannel audio, from which the phase and magnitude components are extracted and used as separate features. The proposed method takes a sequence of consecutive spectrogram frames as input and predicts all the sound event classes active for each of the input frame along with their respective spatial location, producing the temporal activity and DOA trajectory for each sound event class. In particular, a convolutional recurrent neural network (CRNN) is used to map the frame sequence to the two outputs in parallel. At the first output, SED is performed as a multi-label multi-class classification task, allowing the network to simultaneously estimate the presence of multiple sound events for each frame. At the second output, DOA estimates in the continuous 3D space are obtained as a multi-output regression task, where each sound event class is associated with three regressors that estimate the 3D Cartesian coordinates x, y and z of the DOA on a unit sphere around the microphone.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-net/blob/master/images/CRNN_JSTSP.jpg" width="400" title="SELDnet Architecture">
</p>

The SED output of the network is in the continuous range of [0 1] for each sound event in the dataset, and this value is thresholded to obtain a binary decision for the respective sound event activity as shown in figure below. Finally, the respective DOA estimates for these active sound event classes provide their spatial locations.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-net/blob/master/images/JSTSP_output_format.jpg" width="400" title="SELDnet output format">
</p>

The figure below visualizes the SELDnet input and outputs for simulated datasets with maximum one (O1) and two (O2) temporally overlapping sound events. The horizontal-axis of all sub-plots for a given dataset represents the same time frames, the vertical-axis for spectrogram sub-plot represents the frequency bins, vertical-axis for SED reference and prediction sub-plots represents the unique sound event class identifier, and for the DOA reference and prediction sub-plots, it represents the distance from the origin along the respective axes. The 'o' markers in left figure and '•' markers in right figure visualize both the groundtruth labels and predictions of DOA and SED for O1 and O2 datasets. The − markers in the left figure visualizes the results for test data with unseen DOA labels (shifted by 5 degree along azimuth and elevation). The figures represents each sound event class and its associated DOA outputs with a unique color.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-net/blob/master/images/echoic0_ov1_split1_regr3_3d0_19465735_plus5.jpg" width="400" title="dataset with maximum one (O1) temporally overlapping sound events">
   <img src="https://github.com/sharathadavanne/seld-net/blob/master/images/echoic0_ov2_split1_regr3_3d0_19425208.jpg" width="400" title="dataset with maximum two (O2) temporally overlapping sound events">
</p>

## DATASETS
We are releasing all the simulated datasets and the small real-life dataset without ambiance used in the paper on zenodo.org. These datasets are in the range of 30-45 GB and fit within the dataset budget of zenodo.org. The larger datasets can be shared upon request. All of these datasets consist of stationary point sources each associated with a spatial coordinate.

The datasets released are
1. ANSIM (TUT Sound Events 2018 - Ambisonic, Anechoic and Synthetic Impulse Response Dataset) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1237703.svg)](https://doi.org/10.5281/zenodo.1237703)
2. RESIM (TUT Sound Events 2018 - Ambisonic, Reverberant and Synthetic Impulse Response Dataset) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1237707.svg)](https://doi.org/10.5281/zenodo.1237707)
3. CANSIM (TUT Sound Events 2018 - Circular array, Anechoic and Synthetic Impulse Response Dataset) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1237752.svg)](https://doi.org/10.5281/zenodo.1237752)
4. CRESIM (TUT Sound Events 2018 - Circular array, Reverberant and Synthetic Impulse Response Dataset) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1237754.svg)](https://doi.org/10.5281/zenodo.1237754)
5. REAL (TUT Sound Events 2018 - Ambisonic, Reverberant and Real-life Impulse Response Dataset) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1237793.svg)](https://doi.org/10.5281/zenodo.1237793)

All the datasets contain three sub-datasets with maximum one (ov1), two (ov2) and three (ov3) temporally overlapping sound events. Each of these sub-datasets have three cross-validation splits (split1, split2 and split3). In total each dataset has nine splits saved as separate zip files. In order to test the SELDnet code you don't have to download the entire dataset. You can simply download one of the zip files and train the SELDnet for the respective overlap (ov) and split (split).

## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The batch_feature_extraction.py is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The parameter.py script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The cls_feature_class.py script has routines for labels creation, features extraction and normalization.
* The cls_data_generator.py script provides feature + label data in generator mode for training.
* The keras_model.py script implements the SELDnet architecture.
* The evaluation_metrics.py script, implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and the DOA metrics explained in the paper
* The seld.py is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
* The utils.py script has some utility functions.

If you are only interested in the SELDnet model then just check the keras_model.py script.


### Prerequisites

The requirements.txt file consists of the libraries and their versions used. The Python script is written and tested in 2.7.10 version. You can install the requirements by running the following line

```
pip install -r requirements.txt
```


### Training the SELDnet on the datasets
The SELDnet code trains the network for a given overlap (ov1, ov2 or ov3) and split (split1, split2 or split3) at a time. In order to quickly train SELDnet follow the steps below.

* For the chosen dataset, overlap and split, download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are downloading overlap 1 and split 1 of the dataset, then the 'base_folder/' should have two folders - 'wav_ov1_split1_30db/' and 'desc_ov1_split1/' after unzipping.

* Now update the dataset path in cls_feature_class.py script (`self._base_folder = 'base_folder/'`). The normalized features, and labels are written in the same folder, so make sure you have sufficient space for it.

* Update the parameters `overlap` and `split` in parameter.py script. You can now train the SELDnet using default parameters using
```
python seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the parameters.py script and call them as following
```
python seld.py <job-id> <task-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

## License

This repository is licensed under the TUT License - see the [LICENSE](LICENSE.md) file for details

## Acknowledgments

The research leading to these results has received funding from the European Research Council under the European Unions H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND.
