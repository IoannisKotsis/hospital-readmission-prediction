import shap
from preprocessing import data_preprocessing
import pickle
import matplotlib.pyplot as plt


def shap_explainer(model, X_test):

    
    # Initializing a SHAP explainer
    explainer = shap.TreeExplainer(model)

    # SHAP values calculation
    shap_values = explainer.shap_values(X_test)

    # Visualization
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, plot_size=None)
    plt.savefig('shap_values.png', bbox_inches='tight', pad_inches=0.3, dpi=100)
    plt.close()

    return shap_values


if __name__ == "__main__":
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    _, X_test, *_ = data_preprocessing()

    shap_values = shap_explainer(model, X_test)