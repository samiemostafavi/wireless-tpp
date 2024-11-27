# Wireless TPP

<div align="center">
  <a href="PyVersion">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.9+-blue.svg">
  </a>
  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/license-Apache-000000.svg?&color=f5de53">
  </a>
  <a href="commit">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/ant-research/EasyTemporalPointProcess">
  </a>
</div>


`WirelessTPP` is an easy-to-use development and application toolkit for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP) in wireless networks.
<span id='top'/>


## Model List <a href='#top'>[Back to Top]</a>
<span id='model-list'/>

We implement various state-of-the-art TPP papers:

| No  | Publication |     Model     | Paper                                                                                                                                    | Implementation                                                                                                             |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
|  1  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Tensorflow](easy_tpp/model/tf_model/tf_nhp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_nhp.py)                       |
|  2  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Tensorflow](easy_tpp/model/tf_model/tf_fullnn.py)<br/>[Torch](easy_tpp/model/torch_model/torch_fullynn.py)                |
|  3  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Tensorflow](easy_tpp/model/tf_model/tf_sahp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_sahp.py)                     |
|  4  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Tensorflow](easy_tpp/model/tf_model/tf_thp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_thp.py)                       |
|  5  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Tensorflow](easy_tpp/model/tf_model/tf_intensity_free.py)<br/>[Torch](easy_tpp/model/torch_model/torch_intensity_free.py) |



## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We use EDAF and Openairinterface 5G for creation of the dataset

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
git clone https://github.com/samiemostafavi/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
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


 
## Benchmark <a href='#top'>[Back to Top]</a>
<span id='benchmark'/>



## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/ant-research/EasyTPP/blob/master/NOTICE) file for more information.


## Todo List <a href='#top'>[Back to Top]</a>
<span id='todo'/>


## Citation <a href='#top'>[Back to Top]</a>
<span id='citation'/>



## Acknowledgment <a href='#top'>[Back to Top]</a>
<span id='acknowledgment'/>

The project is jointly initiated by Machine Intelligence Group, Alipay and DAMO Academy, Alibaba. 

The following repositories are used in `EasyTPP`, either in close to original form or as an inspiration:

- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)
- [EasyRec](https://github.com/alibaba/EasyRec)
- [EasyNLP](https://github.com/alibaba/EasyNLP)
- [FuxiCTR](https://github.com/xue-pai/FuxiCTR)
- [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)
- [Neural Hawkes Particle Smoothing](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)
- [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
- [Huggingface - transformers](https://github.com/huggingface/transformers)


