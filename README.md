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


## Model List <a href='#top'>[Back to Top]</a>
<span id='model-list'/>

We implement 2 state-of-the-art temporal point process (TPP) papers:

| No  | Publication |     Model     | Paper                                                                                                                                |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
|  1  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           |
|  2  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  |


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

We provide an end-to-end example for users to run a standard TPP model with `WirelessTPP`.


### Step 1. Installation

First of all, we can install the package either by using pip or from the source code on Github.

To install the latest stable version:
```bash
pip install wireless-tpp
```

To install the latest on GitHub:
```bash
git clone https://github.com/samiemostafavi/wireless-tpp.git
cd wireless-tpp
python setup.py install
```


### Step 2. Prepare datasets 

Preprocess edaf files and create database file:
```
python main.py -t preprocess_edaf -s data/s63_results
```

Plot the processed data (packet_arrival)
```
python main.py -t packet_arrival -u plot_data -s data/s63_results -c config/dataset_config.json -g s63 -n test0
```

Plot the processed data (link_quality)
```
python main.py -t link_quality -u plot_data -s data/s63_results -c config/dataset_config.json -g s63 -n test0
python main.py -t link_quality -u plot_data -f -s data/s63_results -c config/dataset_config.json -g s63 -n test0
```

Plot the processed data (scheduling)
```
python main.py -t scheduling -u plot_data -s data/s63_results -c config/dataset_config.json -g s63 -n test0
python main.py -t scheduling -u plot_data -v -s data/s63_results -c config/dataset_config.json -g s63 -n test0
```

Create the dataset (packet arrival)
```
python main.py -t packet_arrival -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_arrival -n test0
```

Create the dataset (link quality retransmissions)
```
python main.py -t link_quality -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_link_retx -n test0
```

Create the dataset (link quality mcs)
```
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_link_mcs -n test0
```

Create the dataset (scheduling)
```
python main.py -t scheduling -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_scheduling -n test0
```


### Step 3. Train the models

First modify the yaml file `config/training_config.yaml` with the datasets information and the configuration of the model you would like to train.

Train a model (packet arrival)
```
python main.py -t packet_arrival -u train_model -f -c config/training_config.yaml -i IF_train_s63_packetarrival_0
```

Train a model (link quality)
```
python main.py -t link_quality -u train_model -f -c config/training_config.yaml -i THP_train_s63_linkquality_0
```

Train a model (scheduling)
```
python main.py -t scheduling -u train_model -f -c config/training_config.yaml -i IF_train_s63_scheduling_0
```

### Step 4. Validate the models

By running the following commands you can check visually if the model is performing or not. 
It will take samples from the test dataset, runs predictions on them, and plots the results.

Validate packet arrival model (probabilistic)
```
python main.py -t packet_arrival -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_arrival -n test0 -i 1105474_140407072232064_241202-120400
python main.py -t packet_arrival -u plot_predictions -s data/s63_results -n test0 -i 1117955_140163946500736_241203-075703
```

Validate packet arrival model (sampling)
```
python main.py -t packet_arrival -u generate_predictions -s data/s63_results -p sampling -c config/prediction_config.json -g s63_arrival -n test0 -i 1105474_140407072232064_241202-120400
python main.py -t packet_arrival -u plot_predictions -s data/s63_results -n test0 -i 1119416_139811246588544_241203-084053
```

Validate retx link quality model (probabilistic)
```
python main.py -t link_quality -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_linkquality -n test0 -i 1106489_139985643180672_241202-123840
python main.py -t link_quality -u plot_predictions -s data/s63_results -n test0 -i 1121178_140591209673344_241203-091730
```

Validate retx link quality model (sampling)
```
python main.py -t link_quality -u generate_predictions -s data/s63_results -p sampling -c config/prediction_config.json -g s63_linkquality -n test0 -i 1105474_140407072232064_241202-120400
python main.py -t link_quality -u plot_predictions -s data/s63_results -n test0 -i 1122926_140662824829568_241203-102211
```

Validate the scheduling model (probabilistic)
```
python main.py -t scheduling -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_scheduling -n test0 -i 1112063_140079729623680_241202-162330
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 1
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 2
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 3
```

Validate the scheduling model (sampling)
```
python main.py -t scheduling -u generate_predictions -s data/s63_results -p sampling -c config/prediction_config.json -g s63_scheduling -n test0 -i 1112063_140079729623680_241202-162330
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1126371_140091332403840_241203-121033 -m 1
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1126371_140091332403840_241203-121033 -m 2
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1126371_140091332403840_241203-121033 -m 3
```

 
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


