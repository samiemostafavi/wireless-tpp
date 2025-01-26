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
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train5k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train10k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train20k -f
python main.py -t e2e -u create_training_dataset -v -s data/intervals_results -c config/dataset_config.json -g intervals_e2e_train -n sub_train50k -f
```


## 2. Train a delay prediction model <a href='#top'>[Back to Top]</a>

First, modify or create the `config/e2e_training_config.yaml` file. Make sure in the data section you have created a subsection for the training datasets you created before according to the example. Next, create sections for the models configurations.

Train a model using the following command and specifying the model section in the yaml file (`-i`).
```
python main.py -t e2e -u train_model -f -c config/training_config.yaml -i e2e_intervals_10k_transformer
```

## 3. Evaluate an e2e model <a href='#top'>[Back to Top]</a>
```
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/e2e_prediction_config.json -g intervals_e2e_eval -n test10k_oa_single_mlp -i 2199105_140148183655040_250120-184431
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/e2e_prediction_config.json -g intervals_e2e_eval -n test10k_oa_40ev_rnn -i 2199105_140148183655040_250120-184431
python main.py -t e2e -u evaluate_model -s data/intervals_results -c config/e2e_prediction_config.json -g intervals_e2e_eval -n test10k_oa_40ev_transformer -i 2199105_140148183655040_250120-184431
```


## 4. Validate a trained scheduling model <a href='#top'>[Back to Top]</a>

Using the following commands, you can visually validate how the predictor works. We plot random instances in the test part of the training dataset or any dataset specified in the configuration json (if the dataset name and id is left empty, we use the training dataset).
Make sure you have an entry in the `scheduling_prediction_config.json` config file with the prediction configurations like number of samples etc.
We can either do PDF prediction or sampling prediction. You can run probabilistic predictions (PDF) using `-p probabilistic` or sample the predictor `-p sampling` in the prediction command below.
The arguments `-n NAME` and `-i ID` are corresponding to the trained model that you desire to validate.
```
python main.py -t scheduling -u generate_predictions -s data/s63_results -p probabilistic -c config/prediction_config.json -g s63_scheduling -n test0 -i 1643939_139725269271168_250108-063415
python main.py -t scheduling -u generate_predictions -s data/multi_size_scheduling -p probabilistic -c config/prediction_config.json -g multi_size_scheduling_eval -n test10k -i 1679990_140223206195840_250108-185841
```
After the predictions are done, we will have in `scheduling/prediction_results/NAME/PREDICTION_ID` folder, all the information and `pred.pkl` file that holds the produced results.

Then we can plot the results using the following commands:
```
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1649824_140555386548864_250108-082849 -m 1
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 2
python main.py -t scheduling -u plot_predictions -s data/s63_results -n test0 -i 1124079_140705547641472_241203-105910 -m 3

python main.py -t scheduling -u plot_predictions -s data/multi_size_scheduling -n test10k -i 1678035_140564617605760_250108-181906 -m 1
python main.py -t scheduling -u plot_predictions -s data/multi_size_scheduling -n test10k -i 1688384_140045867438720_250109-030445 -m 1
```
The argument `-m` corresponds to the segment number if you want to see the prediction for a specific segment.

## 5. Evaluate a trained scheduling model <a href='#top'>[Back to Top]</a>

Simillar to validation, this is how you can evaluate a model against a certain dataset specified in the entry in the config file `scheduling_prediction_config.json`.
Run predictions over the test dataset and produce a json with evaluation metrics.
```
python main.py -t scheduling -u evaluate_model -s data/s63_results -c config/prediction_config.json -g s63_scheduling -n test0 -i 1365509_140428493193856_241228-132954
python main.py -t scheduling -u evaluate_model -s data/s61-64_results -c config/prediction_config.json -g s61-64_scheduling_eval -n test0 -i 1365509_140428493193856_241228-132954
python main.py -t scheduling -u evaluate_model -s data/multi_size_scheduling -c config/prediction_config.json -g multi_size_scheduling_eval -n test10k -i 1672127_140492844757632_250108-170748
```
