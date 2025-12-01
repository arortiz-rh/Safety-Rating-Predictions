import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

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

    model = LogisticRegression(max_iter=800) # The default is 100 iterations, but unless it needed to be higher to stablize weights.
    model.fit(X_training, y_training)

    y_pred = model.predict(X_testing)
    accuracy = accuracy_score(y_testing, y_pred)
    precision = precision_score(y_testing, y_pred)
    f1 = f1_score(y_testing, y_pred)

    print("Metrics\n--------------------------------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("F1: ", f1)

    print("\nFeature Weights\n--------------------------------")
    weights = model.coef_[0]   # array of coefficients
    labels = X_training.columns

    coef_df = pd.DataFrame({"Feature": labels, "Weight": weights})

    coef_df["AbsWeight"] = coef_df["Weight"].abs()
    coef_df = coef_df.sort_values(by="AbsWeight", ascending=False)

    # These weights are from the unscalled model. While most will be accurate, the kerb weights weight will be really small.
    print(coef_df.to_string(index=False))

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

main()