from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

def features_selection(features_path, label, value):
    df = pd.read_csv(features_path)

    new_df = df.loc[df["Type"] == "Train"]

    scaler = LabelEncoder()
    new_df[label] = scaler.fit_transform(new_df[label])

    x_train, y_train = new_df.drop(columns=["Label", "Path", "Type", "Fish Name"]), new_df[label]

    cols = x_train.columns

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    lgbm = LGBMClassifier(objective="multiclass", random_state=42, force_col_wise=True, verbose=-1)
    lgbm.fit(x_train, y_train)

    feature_importance = lgbm.feature_importances_

    feature_importance_df = pd.DataFrame({"Feature": cols, "Importance": feature_importance})
    feature_importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    features = feature_importance_df.loc[feature_importance_df.index < value * len(feature_importance_df)]["Feature"].tolist()

    return features