# E2E Prediction 

This predictor takes a history of previous block transmission events plus packet arrivals, and predicts the delay of the future packets sequeunce. Predictions are all probabilistic.

Create an entry in `config/e2e_dataset_config.json` with a key for example `intervals_e2e_train` to store dataset creation configs.

## 1. Create a packet delay dataset <a href='#top'>[Back to Top]</a>

Create the main dataset (takes time usually). The following commands use the time masks in the dataset config.
Here the code will look up the `time_masks` setting and `only_arrivals`.
Time masks will filter which part of the experiment to extract data from and only_arrivals ignores block transmission attempts and we will only have sequences of arrival events with the `time_since_last_event` representing the delay of the packet.
```
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n main_train
```
Note: for this command, -n “NAME” should match with “main_ds_name: NAME” in the corresponding json entry.

Create training sub-dataset using the same command but with `-f` argument (which randomly selects entries from the main dataset according to the specified size and sequence length). 
This command is fast compared to the previous one.
Set the `split_ratios`, `window_config`, and the number of samples `dataset_size_max` in the dataset config file according to your setting and run:
```
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train1k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train2p5k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train5k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train10k -f
```

You can also pass `-r 2` argument to specify for how many times the random selection of sequences from the main datasets to be done. `-r 2` will create 2 subdatasets, simillar size and profile only 2 different sample sets.
```
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train5k -f -r 2
```
This doesn't work on main dataset creation.


## 2. Train a delay prediction model <a href='#top'>[Back to Top]</a>

First, modify or create the `config/e2e_training_config.yaml` file. Make sure in the data section you have created a subsection for the training datasets you created before according to the example. Next, create sections for the models configurations.

Train a model using the following command and specifying the model section in the yaml file (`-i`).
```
python main.py -t e2e -u train_model -s data/s61-64_results -c config/training_config_s61-64.yaml -i mlp -n 1k
```
Here `-n` specifies the dataset id in the yaml file to train the model with and `-i` sepcifies the model in the config file.

## 3. Evaluate an e2e model <a href='#top'>[Back to Top]</a>

Modify the prediction config json file with your desired entry in it. Then evaluate the trained models using the commands as follows:
```
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/prediction_config.json -g intervals_e2e_eval -n 10k_mlp_50_EXC -i 2199105_140148183655040_250120-184431
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/prediction_config.json -g intervals_e2e_eval -n 10k_lstmsingle_50_EXC -i 2199105_140148183655040_250120-184431
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/prediction_config.json -g intervals_e2e_eval -n 10k_transformer_50_EXC -i 2199105_140148183655040_250120-184431
```
If there is only one model trained, you can ignore `-i` option and only pass the name of the model like:
```
python main.py -t e2e -u evaluate_model -s data/s61-64_results -c config/prediction_config.json -g intervals_e2e_eval -n 10k_transformer_50_EXC
```
The results will be saved in a json file in a new folder with the same name as the trained model in `prediction_results` folder.

## 4. Validate a trained scheduling model <a href='#top'>[Back to Top]</a>


First two figures of the paper are made via:
```
python paper_plot_valid.py -s data/s61-64_results -c config/prediction_config.json -g validate -n 10k_transformer_100_EXC
python paper_plot_valid.py -s data/s61-64_results -c config/prediction_config.json -g validate -n 10k_mlp_100_EXC
```