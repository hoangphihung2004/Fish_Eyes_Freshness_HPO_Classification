import os.path
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
import warnings
import time
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Import feature selection function
sys.path.insert(0, str(Path(__file__).parent.parent / "Feature_Selection"))
from Feature_Selection.feature_selection import features_selection


def _build_pruner(pruner=None):
    """Build Optuna pruner from string or object."""
    if pruner is None:
        return optuna.pruners.HyperbandPruner(min_resource=20, max_resource=100, reduction_factor=3)
    if isinstance(pruner, str):
        key = pruner.strip().lower()
        if key in {"hyperband", "hyperbandpruner"}:
            return optuna.pruners.HyperbandPruner(min_resource=20, max_resource=100, reduction_factor=3)
        if key in {"asha", "successivehalving", "successivehalvingpruner"}:
            return optuna.pruners.SuccessiveHalvingPruner(min_resource=20, reduction_factor=3)
        if key in {"median", "medianpruner"}:
            return optuna.pruners.MedianPruner()
        if key in {"none", "nopruner", "no"}:
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unsupported pruner: {pruner}")
    return pruner


def _build_sampler(sampler=None):
    """Build Optuna sampler from string or object."""
    if sampler is None:
        return optuna.samplers.TPESampler(seed=42)
    if isinstance(sampler, str):
        key = sampler.strip().lower()
        if key in {"tpe", "tpesampler"}:
            return optuna.samplers.TPESampler(seed=42)
        if key in {"random", "randomsampler"}:
            return optuna.samplers.RandomSampler(seed=42)
        if key in {"cmaes", "cmaessampler"}:
            try:
                return optuna.samplers.CmaEsSampler(seed=42)
            except Exception as exc:
                raise ImportError("CMA-ES sampler requires the optional 'cmaes' package.") from exc
        raise ValueError(f"Unsupported sampler: {sampler}")
    return sampler


def _create_study(pruner=None, sampler=None):
    """Create Optuna study with given pruner and sampler."""
    return optuna.create_study(
        direction="maximize",
        pruner=_build_pruner(pruner),
        sampler=_build_sampler(sampler)
    )


def apply_feature_selection(df, features_path, label, selection_threshold):
    """
    Apply feature selection to the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all features
    features_path : str
        Path to CSV file for feature importance calculation
    label : str
        Name of the label column
    selection_threshold : float
        Threshold for feature selection (0.0 to 1.0)
        e.g., 0.5 means keep top 50% of features

    Returns:
    --------
    pd.DataFrame
        Dataframe with only selected features + label + metadata columns
    list
        List of selected feature names
    """
    selected_features = features_selection(features_path, label, selection_threshold)

    # Keep metadata columns along with selected features
    metadata_cols = ["Label", "Path", "Type", "Fish Name"]
    cols_to_keep = [col for col in metadata_cols if col in df.columns] + selected_features

    df_selected = df[cols_to_keep].copy()

    return df_selected, selected_features


def random_forest(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train Random Forest classifier with HPO."""
    start = time.time()

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 501)
        max_depth = trial.suggest_int("max_depth", 2, 50)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=42
        )

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_val)
        acc = accuracy_score(y_val, y_pred)

        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["random_state"] = 42

    training_start = time.time()

    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(x_train, y_train)

    y_pred_val = best_rf.predict(x_val)
    y_pred_test = best_rf.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def svc_classifiers(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train SVM classifier with HPO."""
    start = time.time()

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        C = trial.suggest_float("C", 0.01, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        coef0 = trial.suggest_float("coef0", 1e-5, 1.0, log=True)

        svc = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, random_state=42)
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["random_state"] = 42

    training_start = time.time()

    best_svc = SVC(**best_params)
    best_svc.fit(x_train, y_train)

    y_pred_val = best_svc.predict(x_val)
    y_pred_test = best_svc.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def knn_classifiers(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train KNN classifier with HPO."""
    start = time.time()

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 3, 27, step=2)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        algorithm = trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            algorithm=algorithm
        )

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_val)
        acc = accuracy_score(y_val, y_pred)

        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params

    training_start = time.time()

    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(x_train, y_train)

    y_pred_val = best_knn.predict(x_val)
    y_pred_test = best_knn.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def lr_classifiers(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train Logistic Regression classifier with HPO."""
    start = time.time()

    def objective(trial):
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        C = trial.suggest_float("C", 1e-3, 100, log=True)
        solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        max_iter = trial.suggest_int("max_iter", 300, 1000)

        lr = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["random_state"] = 42

    training_start = time.time()

    best_lr = LogisticRegression(**best_params)
    best_lr.fit(x_train, y_train)

    y_pred_val = best_lr.predict(x_val)
    y_pred_test = best_lr.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def extratree_classifiers(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train Extra Trees classifier with HPO."""
    start = time.time()

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3, 5, 7, 9, 11])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 8, 9, 11])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 11)

        extratree = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
            verbose=0
        )
        extratree.fit(x_train, y_train)
        y_pred = extratree.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["random_state"] = 42

    training_start = time.time()

    best_et = ExtraTreesClassifier(**best_params, verbose=0)
    best_et.fit(x_train, y_train)

    y_pred_val = best_et.predict(x_val)
    y_pred_test = best_et.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def lgbm_classifiers(x_train, x_val, x_test, y_train, y_val, y_test, n_trials, pruner=None, sampler=None):
    """Train LightGBM classifier with HPO."""
    start = time.time()

    def objective(trial):
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
        num_leaves = trial.suggest_int("num_leaves", 15, 50)
        max_depth = trial.suggest_int("max_depth", 2, 50)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        lgbm = LGBMClassifier(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            class_weight=class_weight,
            objective="multiclass",
            force_col_wise=True,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(x_train, y_train)
        y_pred = lgbm.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        return acc

    study = _create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["random_state"] = 42

    training_start = time.time()

    best_lgbm = LGBMClassifier(
        **best_params,
        objective="multiclass",
        force_col_wise=True,
        verbose=-1
    )

    best_lgbm.fit(x_train, y_train)

    y_pred_val = best_lgbm.predict(x_val)
    y_pred_test = best_lgbm.predict(x_test)

    metrics = {
        "Best Params": best_params,
        "Validation Accuracy": accuracy_score(y_val, y_pred_val),
        "Validation Precision": precision_score(y_val, y_pred_val, average="weighted"),
        "Validation Recall": recall_score(y_val, y_pred_val, average="weighted"),
        "Validation F1": f1_score(y_val, y_pred_val, average="weighted"),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, average="weighted"),
        "Test Recall": recall_score(y_test, y_pred_test, average="weighted"),
        "Test F1_score": f1_score(y_test, y_pred_test, average="weighted"),
        "Tuning time": time.time() - start,
        "Training_Time": time.time() - training_start
    }

    return metrics


def modeling_with_feature_selection(
    data_csv_path,
    features_csv_path,
    selection_threshold=1.0,
    output_csv_path=None,
    pruner=None,
    sampler=None
):
    """
    Main function to train ML models with optional feature selection.

    Parameters:
    -----------
    data_csv_path : str
        Path to the main dataset CSV file
    features_csv_path : str
        Path to CSV file used for feature importance calculation
    selection_threshold : float, default=1.0
        Threshold for feature selection (0.0 to 1.0)
        - 1.0 means keep all features (no selection)
        - 0.5 means keep top 50% of features by importance
        - 0.1 means keep top 10% of features
    output_csv_path : str, optional
        Path to save results CSV. If None, results are not saved
    pruner : str or object, optional
        Optuna pruner: "hyperband", "asha", "median", "none", or object
    sampler : str or object, optional
        Optuna sampler: "tpe", "random", "cmaes", or object

    Returns:
    --------
    pd.DataFrame
        Results dataframe with metrics for each model
    list
        List of selected feature names
    """

    # Validate selection_threshold
    if not (0.0 < selection_threshold <= 1.0):
        raise ValueError("selection_threshold must be between 0.0 (exclusive) and 1.0 (inclusive)")

    fea_name = f"FS_{int(selection_threshold * 100)}%" if selection_threshold < 1.0 else "Full Features"
    label = "Label"

    # Load data
    df = pd.read_csv(data_csv_path)

    # Apply feature selection if threshold < 1.0
    if selection_threshold < 1.0:
        print(f"\nApplying feature selection with threshold {selection_threshold}...")
        df, selected_features = apply_feature_selection(df, features_csv_path, label, selection_threshold)
        print(f"Selected {len(selected_features)} features from {len(selected_features)} available")
    else:
        selected_features = [col for col in df.columns
                            if col not in ["Label", "Path", "Type", "Fish Name"]]

    # Split into train/val/test
    train = df.loc[df["Type"] == "Train"]
    val = df.loc[df["Type"] == "Validation"]
    test = df.loc[df["Type"] == "Test"]

    # Encode labels
    label_encoder = LabelEncoder()
    train[label] = label_encoder.fit_transform(train[label])
    val[label] = label_encoder.transform(val[label])
    test[label] = label_encoder.transform(test[label])

    # Prepare features
    x_train, y_train = train.drop(columns=["Label", "Path", "Type", "Fish Name"]), train[label]
    x_val, y_val = val.drop(columns=["Label", "Path", "Type", "Fish Name"]), val[label]
    x_test, y_test = test.drop(columns=["Label", "Path", "Type", "Fish Name"]), test[label]

    # Standardize features
    scaler_ = StandardScaler()
    x_train = scaler_.fit_transform(x_train)
    x_val = scaler_.transform(x_val)
    x_test = scaler_.transform(x_test)

    # Initialize result dictionary
    result = {
        "Model": [],
        "Feature Name": [],
        "Validation Accuracy": [],
        "Validation Precision": [],
        "Validation Recall": [],
        "Validation F1": [],
        "Test Accuracy": [],
        "Test Precision": [],
        "Test Recall": [],
        "Test F1_score": [],
        "Tuning time": [],
        "Best Param": [],
        "Training_Time": []
    }

    # Define models and number of trials
    models = {
        "KNN": 150,
        "LR": 30,
        "SVM": 90,
        "RF": 100,
        "LGBM": 100,
        "EXTree": 100,
    }

    print(f"\nTraining models with {fea_name}...")

    # Train each model
    for model_name, n_trials in models.items():
        print(f"  Training {model_name}...")

        if model_name == "KNN":
            metrics = knn_classifiers(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        elif model_name == "LR":
            metrics = lr_classifiers(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        elif model_name == "LGBM":
            metrics = lgbm_classifiers(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        elif model_name == "EXTree":
            metrics = extratree_classifiers(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        elif model_name == "RF":
            metrics = random_forest(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        elif model_name == "SVM":
            metrics = svc_classifiers(
                x_train.copy(), x_val.copy(), x_test.copy(),
                y_train.copy(), y_val.copy(), y_test.copy(),
                n_trials, pruner=pruner, sampler=sampler
            )
        else:
            raise ValueError(f"Model {model_name} not found")

        # Collect results
        result["Model"].append(model_name)
        result["Feature Name"].append(fea_name)
        result["Validation Accuracy"].append(metrics["Validation Accuracy"])
        result["Validation Precision"].append(metrics["Validation Precision"])
        result["Validation Recall"].append(metrics["Validation Recall"])
        result["Validation F1"].append(metrics["Validation F1"])
        result["Test Accuracy"].append(metrics["Test Accuracy"])
        result["Test Precision"].append(metrics["Test Precision"])
        result["Test Recall"].append(metrics["Test Recall"])
        result["Test F1_score"].append(metrics["Test F1_score"])
        result["Tuning time"].append(metrics["Tuning time"])
        result["Best Param"].append(metrics["Best Params"])
        result["Training_Time"].append(metrics["Training_Time"])

    results_df = pd.DataFrame(result)

    # Save results if output path provided
    if output_csv_path:
        if os.path.exists(output_csv_path):
            results_df.to_csv(output_csv_path, mode="a", header=False, index=False)
            print(f"\nResults appended to {output_csv_path}")
        else:
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nResults saved to {output_csv_path}")

    return results_df, selected_features


if __name__ == "__main__":
    # Example usage
    # Set your paths here
    data_csv_path = r""  # Path to main dataset
    features_csv_path = r""  # Path to features data for importance calculation
    output_csv_path = r""  # Path to save results

    # Run with feature selection (keep top 50% features)
    results, selected_features = modeling_with_feature_selection(
        data_csv_path=data_csv_path,
        features_csv_path=features_csv_path,
        selection_threshold=0.5,  # Keep top 50% of features
        output_csv_path=output_csv_path,
        pruner="hyperband",
        sampler="tpe"
    )

    print("\nResults Summary:")
    print(results)
    print(f"\nSelected {len(selected_features)} features:")
    print(selected_features)

