# Link Quality Prediction

## MCS Prediction <a href='#top'>[Back to Top]</a>

### 1. Check the MCS data

### 2. Create an MCS dataset

Create the dataset (link quality mcs)
```
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n main_mcs_eval
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n sub_mcs_eval -f
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n sub_mcs_train5k-h100 -f 
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n sub_mcs_train10k-h100 -f 
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n sub_mcs_train20k-h100 -f
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_mcs_eval -n sub_mcs_train20k-h200 -f 

python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/dataset_config.json -g s61-64_mcs -n mcs0
python main.py -t link_quality -u create_training_dataset -s data/s63_results -c config/dataset_config.json -g s63_mcs -n mcs0
```

### 3. Train an MCS model

Train a model (link quality)
```
python main.py -t link_quality -u train_model -f -c config/linkquality_training_config.yaml -i MCS_s61-64_20k
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
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h5_noisy -i 1862244_139922419221120_250112-065712
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h5_noisy_head2 -i 1870587_139808944210560_250112-132935

python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h10_noisy -i 1862388_140188890497664_250112-065735
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h10_noisy_head2 -i 1870741_139836598248064_250112-132956

python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h50_noisy -i 1862551_139848559821440_250112-065756
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h50_noisy_head2 -i 1870919_140295854105216_250112-133033

python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h100_noisy -i 1858263_139847514428032_250112-053700
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h100_noisy_head2 -i 1863554_139962978595456_250112-070158

python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h200_noisy -i 1871312_140079486190208_250112-133154
python main.py -t link_quality -u evaluate_model -s data/s61-64_results -c config/linkquality_prediction_config.json -g s61-64_mcs_eval -n mcs_20k_h200_noisy_head2 -i 1871104_140120553767552_250112-133115
```

## RETX Prediction <a href='#top'>[Back to Top]</a>

Modify the file `linkquality_dataset_config.json` and make sure there is a configuration for retx dataset creation.

### 1. Check the RETX data

Plot the processed data
```
python main.py -t link_quality -u plot_data -s data/s63_results -c config/linkquality_dataset_config.json -g s63_link_retx -n test0
python main.py -t link_quality -u plot_data -v -s data/s63_results -c config/linkquality_dataset_config.json -g s63_link_retx -n test0
python main.py -t link_quality -u plot_data -f -s data/s63_results -c config/linkquality_dataset_config.json -g s63_link_retx -n test0
```


### 2. Create a RETX dataset

Create the dataset
```
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_retx_eval -n main_retx_eval
python main.py -t link_quality -u create_training_dataset -s data/s63_results -c config/linkquality_dataset_config.json -g s63_retx -n retx0
python main.py -t link_quality -u create_training_dataset -s data/s61-64_results -c config/linkquality_dataset_config.json -g s61-64_retx_eval -n sub_eval0 -f
```

Calculate retx probabilities
```
python main.py -t link_quality -u plot_data -v -s data/s61-64_results -c config/linkquality_dataset_config.json -g multi_size_retx_eval -n main_retx_eval
python main.py -t link_quality -u plot_data -v -s data/s61-64_results -c config/linkquality_dataset_config.json -g multi_size_mcs_eval -n main_mcs_eval
```


```
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_train -n main_train
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_train -n sub_train5k_h5 -f
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_train -n sub_train10k_h5 -f
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_train -n sub_train20k_h5 -f
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_train -n sub_train50k_h5 -f
```
```
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_eval -n main_eval
python main.py -t link_quality -u create_training_dataset -s data/multi_size_scheduling -c config/linkquality_dataset_config.json -g multi_size_retx_eval -n sub_eval0 -f
```

### 3. Train a RETX model

Train a model
```
python main.py -t link_quality -u train_model -f -c config/linkquality_training_config.yaml -i RETX_multi_size_5k
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
python main.py -t link_quality -u evaluate_model -s data/multi_size_scheduling -c config/linkquality_prediction_config.json -g multi_size_retx_eval -n test50k-cond -i 1777322_140262763119232_250110-144350
python main.py -t link_quality -u evaluate_model -s data/multi_size_scheduling -c config/linkquality_prediction_config.json -g multi_size_retx_eval -n test20k-cond -i 1775994_140351028277888_250110-143104
python main.py -t link_quality -u evaluate_model -s data/multi_size_scheduling -c config/linkquality_prediction_config.json -g multi_size_retx_eval -n test10k-cond -i 1776616_140200386671232_250110-143704
python main.py -t link_quality -u evaluate_model -s data/multi_size_scheduling -c config/linkquality_prediction_config.json -g multi_size_retx_eval -n test5k-cond -i 1770738_140582421906048_250110-123404
```