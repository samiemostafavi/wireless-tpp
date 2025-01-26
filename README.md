# WirelessTPP

<div align="center">
  <a href="PyVersion">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.9+-blue.svg">
  </a>
  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/license-Apache-000000.svg?&color=f5de53">
  </a>
  <a href="commit">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/samiemostafavi/wireless-tpp">
  </a>
</div>

<span id='top'/>

`WirelessTPP` is a development toolkit for [Mixture Density Networks](https://reference.wolfram.com/language/tutorial/NeuralNetworksRegressionWithUncertainty.html) (MDN) and [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP) for temporal performance prediction in Openairinterface5G.

We use EDAF as a dependent project which collects the data from all over the 5G network.


## Model List <a href='#top'>[Back to Top]</a>
<span id='model-list'/>

We implement 2 state-of-the-art probabilistic time series prediction methods:
- LSTM
- Transformer


## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We use EDAF and Openairinterface 5G for creation of the dataset.

## Quick Start <a href='#top'>[Back to Top]</a>
<span id='quick-start'/>

This code is tested with Python 3.9. 
To create a Python 3.9 environment with Conda, you can use the following command:

```shell
conda create --name wireless_tpp python=3.9
```
This command will create a new Conda environment with Python 3.9 installed.

```shell
conda activate wireless_tpp
```

We provide an end-to-end example for users to run a standard model with `WirelessTPP`.


### Step 1. Installation

First of all, we can install the package either by using pip or from the source code on Github.

To install the latest version from GitHub:
```bash
git clone https://github.com/samiemostafavi/wireless-tpp.git
cd wireless-tpp
git checkout develop
python setup.py install
```


### Step 2. Create the datasets from EDAF files

Place EDAF folders and the experiment conf file in a folder (source folder). Preprocess edaf files and create database file using:
```
python main.py -t preprocess_edaf -s data/s63_results
```
In addition to the database files, there should be a json containing the experiment parameters: `experiment_config.json`.


### Step 3. Train a Model and Predict

Refer to `DELAY_PREDICTION.md' file.
 
## Benchmark <a href='#top'>[Back to Top]</a>
<span id='benchmark'/>



## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/samiemostafavi/wireless-tpp/blob/main/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/samiemostafavi/wireless-tpp/blob/main/NOTICE) file for more information.


## Todo List <a href='#top'>[Back to Top]</a>
<span id='todo'/>


## Citation <a href='#top'>[Back to Top]</a>
<span id='citation'/>



## Acknowledgment <a href='#top'>[Back to Top]</a>
<span id='acknowledgment'/>

The following repositories are used in `WirelessTPP`, either in close to original form or as an inspiration:

- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)
- [Huggingface - transformers](https://github.com/huggingface/transformers)


