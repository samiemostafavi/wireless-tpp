# Scheduling Prediction 

This predictor takes a history of previous scheduling events plus packet arrivals, and predicts the time and size of the next schedule. Predictions are all probabilistic.

Create an entry in `config/scheduling_dataset_config.json` with a key for example `s63_scheduling` to store dataset creation configs.

## 1. Check the scheduling data <a href='#top'>[Back to Top]</a>

Plot the processed data, use `-v` to plot interarrival plot
```
python main.py -t scheduling -u plot_data -s data/s63_results -c config/scheduling_dataset_config.json -g s63_scheduling -n test0
python main.py -t scheduling -u plot_data -v -s data/s63_results -c config/scheduling_dataset_config.json -g s63_scheduling -n test0
```

## 2. Create an scheduling dataset <a href='#top'>[Back to Top]</a>


Create the main dataset (takes time usually). The following commands use the time masks in the dataset config.
```
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling -n main_train
```
```
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_train -n main_train
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_eval -n main_eval
```

Create training sub-dataset (which selects entries from the main dataset according to the specified size randomly) and it is fast. Make sure change the `dataset_size_max` setting in the dataset config file.
```
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_train -n sub_train5k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_train -n sub_train10k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_train -n sub_train20k -f
python main.py -t scheduling -u create_training_dataset -s data/multi_size_scheduling -c config/scheduling_dataset_config.json -g multi_size_scheduling_eval -n sub_eval -f
```
```
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling_train -n sub_train20k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling_train -n sub_train10k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling_train -n sub_train5k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling_train -n sub_train2p5k -f
python main.py -t scheduling -u create_training_dataset -s data/s61-64_results -c config/scheduling_dataset_config.json -g s61-64_scheduling_eval -n sub_eval0 -f
```


## 3. Train an scheduling model <a href='#top'>[Back to Top]</a>

Train a model. Here we use a transformer MDN (Encoder only) model with 1 layer and 2 heads by default and the following settings:
* Hidden size: 16
* Total number of parameters: 3000
* Number of epochs: 500
* Batch size: 64 (hyperparameter tuning showed that using small batch sizes are really important)
* Training durations: ~ 0.5 to 2 hours per model
Then we change for each training: 
* History length (number of events in the history): 5, 10, 20
* Event embedding: base and extended (base means we only use time and size as for the attributes of history events, extended means number of retransmissions and possible failure of the attempt is also included).
Modify the scheduling_training_config.yaml file with the model configuration and dataset that you want to use for trianing.
```
python main.py -t scheduling -u train_model -f -c config/scheduling_training_config.yaml -i Scheduling_s63_0
python main.py -t scheduling -u train_model -f -c config/scheduling_training_config.yaml -i Scheduling_s61-64_0
python main.py -t scheduling -u train_model -f -c config/scheduling_training_config.yaml -i Scheduling_multi_size_10k
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