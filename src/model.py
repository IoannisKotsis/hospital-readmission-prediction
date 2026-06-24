# Importing packages
from preprocessing import data_preprocessing
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
    )
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import matplotlib.pyplot as plt
import pickle



pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


def tuning(X_train, Y_train):


    # Model parameters distribution
    param_distributions = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [3, 5, 6, 7],
        "learning_rate" : [0.05, 0.1],
        "colsample_bytree" : [0.7, 0.8, 0.9],
        "subsample" : [0.7, 0.8, 0.9],
        "reg_alpha" : [0.1, 0.2],
        "reg_lambda" : [0.1, 0.2],
        "gamma" : [0.15, 0.2],
    }

    # Cross validation item
    ## Returns an object that knows how to split the data
    skf_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 256)

    # Cross validation with random parameters
    ## Creates a search object with all the execution details
    search = RandomizedSearchCV(
        estimator = XGBClassifier(
            scale_pos_weight = (len(Y_train) - sum(Y_train))/sum(Y_train),
            random_state = 33,
            objective = "binary:logistic"
        ),
        param_distributions =  param_distributions,
        n_iter = 150,
        cv = skf_cv,
        scoring = "roc_auc",
        n_jobs = 1
    )


    ## For each parameter combination, it runs 5-fold CV and calculates the mean AUC of them
    search.fit(X_train, Y_train)

    print(f"Best parameters: \n {search.best_params_}")
    return search.best_params_



def final_training(X_train, X_test, Y_train, Y_test, best_params):

    
    model = XGBClassifier (
            **best_params, 
            scale_pos_weight = (len(Y_train) - sum(Y_train))/sum(Y_train),
            random_state = 37,
            objective = "binary:logistic"
            )

    # Model training
    model.fit(X_train, Y_train)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


    # TESTING:

    # Model predictions
    pred = model.predict(X_test)

    # Model_probabilities
    pred_prob = model.predict_proba(X_test)

    # Model probabilities of belonging to the positive class
    pos_class_prob = pred_prob[:, 1]



    # Accuracy
    accuracy = accuracy_score(Y_test, pred)
    print("Accuracy:", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(Y_test, pred)
    print("Conf. matrix:", cm)

    # Precision
    precision = precision_score(Y_test, pred)
    print("Precision:", precision*100)

    # Recall
    recall = recall_score(Y_test, pred)
    print("Recall:", recall*100)

    # F1-Score
    F1_score = f1_score(Y_test, pred)
    print("F1-Score:", F1_score)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(Y_test, pos_class_prob)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("FPR (False Positive Rate)", color = "blue")
    plt.ylabel("TPR (True Positive Rate)", color = "blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # AUC Curve
    auc = roc_auc_score(Y_test, pos_class_prob)
    print("AUC:", auc)

    return model, accuracy, cm, precision, recall, F1_score, auc


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, ode = data_preprocessing()
    best_params = tuning(X_train, Y_train)
    final_training(X_train, X_test, Y_train, Y_test, best_params)