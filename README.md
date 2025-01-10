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


### Step 2. Create the datasets from EDAF files

Place EDAF folders and the experiment conf file in a folder (source folder). Preprocess edaf files and create database file using:
```
python main.py -t preprocess_edaf -s data/s63_results
```

## Packet Arrival Prediction <a href='#top'>[Back to Top]</a>

Create an entry in `config/dataset_config.json` with a key for example `s63_arrival` to store dataset creation configs.

### 1. Check the packet arrivals data

To check packet arrival process, use the following command that produces a figure from the records in the database.
```
python main.py -t packet_arrival -u plot_data -s data/s63_results -c config/dataset_config.json -g s63_arrival -n test0
```

### 2. Create an arrival dataset

Create a dataset for training a packet arrival model.
```
python main.py -t packet_arrival -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_arrival -n test0 -k
```
Use `-k` to avoid event type to length mapping.

### 3. Train an arrival model

First modify the yaml file `config/training_config.yaml` with the datasets information and the configuration of the model you would like to train.

Train a model using
```
python main.py -t packet_arrival -u train_model -f -c config/training_config.yaml -i Arrival_s63_0
```

### 4. Validate a trained arrival model

By running the following commands you can see in a figure if the model is performing or not. 
We run predictions over the entire test dataset:
```
python main.py -t packet_arrival -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_arrival -n test0 -i 1105474_140407072232064_241202-120400
```
You can run probabilistic predictions (PDF) using `-p probabilistic` or sample the predictor `-p sampling`

Check the predictions
```
python main.py -t packet_arrival -u plot_predictions -s data/s63_results -n test0 -i 1117955_140163946500736_241203-075703
```

### 5. Evaluate a trained arrival model

Run predictions over the test dataset and produce a json with evaluation metrics.
```
```

## Scheduling Prediction <a href='#top'>[Back to Top]</a>

Create an entry in `config/dataset_config.json` with a key for example `s63_scheduling` to store dataset creation configs.

### 1. Check the scheduling data

Plot the processed data, use `-v` to plot interarrival plot
```
python main.py -t scheduling -u plot_data -s data/s63_results -c config/dataset_config.json -g s63_scheduling -n test0
python main.py -t scheduling -u plot_data -v -s data/s63_results -c config/dataset_config.json -g s63_scheduling -n test0
```

### 2. Create an scheduling dataset

Create the main dataset (takes time usually)
```
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n main
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/dataset_config.json -g multi_size_scheduling -n main
```

Create training sub-dataset (which selects entries from the main dataset according to the specified size randomly) and it is fast.
```
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n sub_train20k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n sub_train10k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n sub_train5k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n sub_train2p5k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_scheduling -n sub_eval0 -f

python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/dataset_config.json -g multi_size_scheduling -n sub_train5k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/dataset_config.json -g multi_size_scheduling -n sub_train10k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/dataset_config.json -g multi_size_scheduling -n sub_train20k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/dataset_config.json -g multi_size_scheduling -n sub_eval0 -f
```


### 3. Train an scheduling model

Train a model
```
python main.py -t scheduling -u train_model -f -c config/training_config.yaml -i Scheduling_s63_0
python main.py -t scheduling -u train_model -f -c config/training_config.yaml -i Scheduling_s61-64_0
python main.py -t scheduling -u train_model -f -c config/training_config.yaml -i Scheduling_multi_size_10k
```

### 4. Validate a trained scheduling model

Validate the scheduling model
```
python main.py -t scheduling -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_scheduling -n test0 -i 1643939_139725269271168_250108-063415
python main.py -t scheduling -u generate_predictions -s data/multi_size_scheduling -p probabilistic -c config/prediction_config.json -g multi_size_scheduling_eval -n test10k -i 1679990_140223206195840_250108-185841

```
You can run probabilistic predictions (PDF) using `-p probabilistic` or sample the predictor `-p sampling`

Check the predictions
```
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1649824_140555386548864_250108-082849 -m 1
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 2
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 3

python main.py -t scheduling -u plot_predictions -s data/multi_size_scheduling -n test10k -i 1678035_140564617605760_250108-181906 -m 1
python main.py -t scheduling -u plot_predictions -s data/multi_size_scheduling -n test10k -i 1688384_140045867438720_250109-030445 -m 1
```

### 5. Evaluate a trained scheduling model

Run predictions over the test dataset and produce a json with evaluation metrics.
```
python main.py -t scheduling -u evaluate_model -s data/s63_results -c config/prediction_config.json -g s63_scheduling -n test0 -i 1365509_140428493193856_241228-132954
python main.py -t scheduling -u evaluate_model -s data/s61-64_results -c config/prediction_config.json -g s61-64_scheduling_eval -n test0 -i 1365509_140428493193856_241228-132954
python main.py -t scheduling -u evaluate_model -s data/multi_size_scheduling -c config/prediction_config.json -g multi_size_scheduling_eval -n test10k -i 1672127_140492844757632_250108-170748
```

## MCS Prediction <a href='#top'>[Back to Top]</a>

### 1. Check the MCS data

### 2. Create an MCS dataset

Create the dataset (link quality mcs)
```
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_mcs -n mcs0
python main.py -t link_quality -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_mcs -n mcs0
```

### 3. Train an MCS model

Train a model (link quality)
```
python main.py -t link_quality -u train_model -f -c config/training_config.yaml -i MCS_s63_0
```

### 4. Validate a trained MCS model

Validate mcs model
```
python main.py -t link_quality -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_mcs -n mcs0 -i 1106489_139985643180672_241202-123840
```
You can run probabilistic predictions (PDF) using `-p probabilistic` or sample the predictor `-p sampling`

```
python main.py -t link_quality -u plot_predictions -s data/s63_results -n mcs0 -i 1121178_140591209673344_241203-091730
```

### 5. Evaluate a trained MCS model

Run predictions over the test dataset and produce a json with evaluation metrics.
```
```

## RETX Prediction <a href='#top'>[Back to Top]</a>

### 1. Check the RETX data

Plot the processed data
```
python main.py -t link_quality -u plot_data -s data/s63_results -c config/dataset_config.json -g s63_link_retx -n test0
python main.py -t link_quality -u plot_data -v -s data/s63_results -c config/dataset_config.json -g s63_link_retx -n test0
python main.py -t link_quality -u plot_data -f -s data/s63_results -c config/dataset_config.json -g s63_link_retx -n test0
```

### 2. Create a RETX dataset

Create the dataset
```
python main.py -t link_quality -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_retx -n retx0
```

### 3. Train a RETX model

Train a model
```
python main.py -t link_quality -u train_model -f -c config/training_config.yaml -i RETX_s63_0
```

### 4. Validate a trained RETX model

Validate RETX model
```
python main.py -t link_quality -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_retx -n retx0 -i 1457673_140246213436032_250101-060222
```

Plot predictions
```
python main.py -t link_quality -u plot_predictions -s data/s63_results -n retx0 -i 1468033_140013680992896_250101-124139
```

### 5. Evaluate a trained RETX model

Run predictions over the test dataset and produce a json with evaluation metrics.
```
```



## Modular Packet Delay Prediction <a href='#top'>[Back to Top]</a>

Packet delay prediction algorithm:
1) Predict next packet arrival time and size (use packet arrival model)
2) Predict segment 1 scheduling time and number of resource blocks
3) Predict retransmission probability and types for segment 1, and the departure time of it
4) Repeat 2 and 3 for next segments until sum of departed segments sizes becomes greater than packet size

Validate by expected values
```
python main.py -t e2e -u plot_data -s data/s63_results -n test1 -c config/e2e_config.json
python main.py -t e2e -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_e2e -n test1
python main.py -t e2e -u generate_predictions -s data/s63_results -c config/prediction_config.json -g s63_e2e -n test0
python main.py -t e2e -u evaluate_model -s data/s63_results -n test0 -i 1606662_140466152673920_250107-064955
python main.py -t e2e -u plot_predictions -s data/s63_results -n test0 -i 1606662_140466152673920_250107-064955
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


