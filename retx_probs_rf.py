import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def load_data_as_classification(json_path):
    """
    Reads the JSON and returns a DataFrame with columns:
        mcs, rb, label, weight
    where 'label' is one of {0, 1, 2, 3, 4},
    and 'weight' is how many times that outcome occurred.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for mcs_str, rb_dict in data.items():
        mcs = int(mcs_str)
        for rb_str, stats in rb_dict.items():
            rb = int(rb_str)
            retx = stats.get('retx', [0,0,0,0])  # [n0, n1, n2, n3]
            failed = stats.get('failed', 0)
            
            # 0 retx => label=0, 1 retx => label=1, etc.
            for label_val, count_val in enumerate(retx):
                if count_val > 0:
                    rows.append({
                        'mcs': mcs,
                        'rb': rb,
                        'label': label_val,      # 0..3
                        'weight': count_val
                    })
            # 'fail' => label=4
            if failed > 0:
                rows.append({
                    'mcs': mcs,
                    'rb': rb,
                    'label': 4,               # 4 => fail
                    'weight': failed
                })
    df = pd.DataFrame(rows, columns=['mcs', 'rb', 'label', 'weight'])
    return df


def train_rf_classifier(df,unique_labels):
    """
    Trains a RandomForestClassifier to predict label in {0,1,2,3,4}
    from features (mcs, rb). Uses the 'weight' column as sample_weight.
    Returns the trained model.
    """
    X = df[['mcs', 'rb']]
    y = df['label']
    w = df['weight']

    # Train-test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.1, random_state=42, #stratify=y
    )
    
    # Create and train the classifier
    # NOTE: For probability outputs, we need classifiers that support predict_proba
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluate using log-loss (or something that uses probabilities)
    y_pred_proba = clf.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba, sample_weight=w_test, labels=unique_labels)
    print(f"Log-loss on test = {loss:.4f}")
    
    return clf


def predict_probabilities(clf, mcs, rb, unique_labels):
    """
    Given a trained random forest classifier, an MCS value, and a number of RB,
    returns a dictionary of probabilities for {0 retx, 1 retx, 2 retx, 3 retx, fail}.
    Sum of these probabilities is guaranteed to be 1.
    """
    # Build a DataFrame with the same columns as training
    X_new = pd.DataFrame({'mcs':[mcs], 'rb':[rb]})
    # Get probability distribution
    proba = clf.predict_proba(X_new)[0]  # shape = (5,)

    result = {}
    for prob,label in zip(proba,unique_labels):
        result[int(label)] = prob
    
    return result


if __name__ == "__main__":
    json_file = "./data/s61-64_results/link_quality/datasets/main_eval/retx_stats.json"  # your JSON path
    df_classif = load_data_as_classification(json_file)
    unique_labels = df_classif["label"].unique()
    
    clf_model = train_rf_classifier(df_classif, unique_labels)

    # Example of predicting for MCS=16, RB=20
    result = predict_probabilities(clf_model, 14, 5, unique_labels)
    print(result)
