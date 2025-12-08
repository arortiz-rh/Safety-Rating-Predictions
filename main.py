import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def main():
    # Loading data from files
    training_data = pd.read_csv("./data/euro-ncap-data-training.txt")
    testing_data = pd.read_csv("./data/euro-ncap-data-testing.txt")

    multi_bit_columns = ["Center Airbag", "Side Pelvis Airbag", "Side Chest Airbag", "Belt Pretensioner", "Knee Airbag"]

    training_data = split_columns(multi_bit_columns, training_data)
    testing_data = split_columns(multi_bit_columns, testing_data)

    X_training = training_data.drop("Star", axis=1)
    y_training = training_data["Star"]
    X_testing = testing_data.drop("Star", axis=1)
    y_testing = testing_data["Star"]

    # An unbalanced model is balance to converage on weights within 800 iterations.
    # A balanced model requires more than 800 iterations.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_training, y_training)

    y_pred_unscaled = model.predict(X_testing)
    accuracy_unscaled = accuracy_score(y_testing, y_pred_unscaled)
    precision_unscaled = precision_score(y_testing, y_pred_unscaled)
    recall_unscaled = recall_score(y_testing, y_pred_unscaled)
    f1_unscaled = f1_score(y_testing, y_pred_unscaled)

    print("\nUNSCALED MODEL\n")
    produce_model_output(accuracy_unscaled, precision_unscaled, recall_unscaled, f1_unscaled, model, X_training)

    # This helps look at what kind of predicitions the model got right and wrong.
    print("\nConfusion Matrix\n--------------------------------")
    print_confusion_matrix(y_testing, y_pred_unscaled)

    print("\nSCALED MODEL\n")
    scaler = StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_testing_scaled = scaler.transform(X_testing)
    
    model.fit(X_training_scaled, y_training)
    
    y_pred_scaled = model.predict(X_testing_scaled)
    accuracy_scaled = accuracy_score(y_testing, y_pred_scaled)
    precision_scaled = precision_score(y_testing, y_pred_scaled)
    recall_scaled = recall_score(y_testing, y_pred_scaled)
    f1_scaled = f1_score(y_testing, y_pred_scaled)

    produce_model_output(accuracy_scaled, precision_scaled, recall_scaled, f1_scaled, model, X_training)

    print("\nConfusion Matrix\n--------------------------------")
    print_confusion_matrix(y_testing, y_pred_scaled)

# Present because when collecting the data, I made it so that there were strings of 3 integer values (example: -1-1-1), but upon testing
# this doesn't work for Logistic Regression which requires fully numeric values.
def split_columns(columns, data):
    for col in columns:
        split = data[col].astype(str).str.replace("-", "").apply(list)
        split_df = pd.DataFrame(split.tolist(), index=data.index)

        split_df = split_df.astype(int)
        split_df.columns = [f"{col}_Driver", f"{col}_Passenger", f"{col}_Rear"]

        data = pd.concat([data.drop(columns=[col]), split_df], axis=1)
    
    return data

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    print("\nMatrix:")
    print("                Predicted 0    Predicted 1")
    print(f"Actual 0        {tn:12d} {fp:12d}")
    print(f"Actual 1        {fn:12d} {tp:12d}")

def produce_model_output(accuracy, precision, recall, f1, model, X_training):
    print("Metrics\n--------------------------------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    print("\nFeature Weights\n--------------------------------")
    weights = model.coef_[0] # array of coefficients
    labels = X_training.columns

    coef_df = pd.DataFrame({"Feature": labels, "Weight": weights})

    coef_df["AbsWeight"] = coef_df["Weight"].abs()
    coef_df = coef_df.sort_values(by="AbsWeight", ascending=False)

    print(coef_df.to_string(index=False))

main()