import shap
from preprocessing import data_preprocessing
import pickle
import matplotlib.pyplot as plt


def shap_explainer(model, X_test):

    
    #initializing a SHAP explainer
    explainer = shap.TreeExplainer(model)

    #SHAP values calculation
    shap_values = explainer.shap_values(X_test)

    #visualization
    shap.summary_plot(shap_values, X_test, plot_type="dot", show = False, plot_size = (9,7))
    plt.savefig('shap_values.png', bbox_inches = 'tight', pad_inches = 0.3)
    plt.close()

    return shap_values


if __name__ == "__main__":
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    _, X_test, *_ = data_preprocessing()

    shap_values = shap_explainer(model, X_test)