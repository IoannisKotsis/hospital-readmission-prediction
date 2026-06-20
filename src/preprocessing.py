#packages imported for preprocessing
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pickle

def map_icd9(code):

        if pd.isna(code):
            return "Other"
        integer_part = code.split('.')[0]
        try:
            n = int(integer_part)
        except ValueError:
            return "Other"
    
        if 390 <= n <= 459 or n == 785:
            return "Circulatory"
        elif 460 <= n <= 519 or n == 786:
            return "Respiratory"
        elif n == 250:
            return "Diabetes"
        elif 520 <= n <= 579 or n == 787:
            return 'Digestive'
        elif 800 <= n <= 999:
            return 'Injury'
        elif 710 <= n <= 739:
            return 'Musculoskeletal'
        elif 580 <= n <= 629 or n == 788:
            return 'Genitourinary'
        elif 140 <= n <= 239:
            return 'Neoplasms'
        else:
            return "Other"


#function that returns the preprocessed data (X_train, X_test, Y_train, Y_test) for model training and evaluation
def data_preprocessing():


    #dataframe with relative path to csv file
    df = pd.read_csv("../data/diabetic_data.csv", na_values = "?", low_memory = False)

    #renaming two specific columns for SQL naming compatibility
    df = df.rename(columns = {
        'change' : 'med_change',
        'diabetesMed' : 'diabetes_med',
        'glyburide-metformin' : 'glyburide_metformin',
        'glipizide-metformin' : 'glipizide_metformin',
        'glimepiride-pioglitazone' : 'glimepiride_pioglitazone',
        'metformin-rosiglitazone' : 'metformin_rosiglitazone',
        'metformin-pioglitazone' : 'metformin_pioglitazone'
    })

    
    #removes duplicates based on patient_nbr column
    df.drop_duplicates(subset="patient_nbr", inplace = True)

    #removal of rows where any of the diagnosis columns have missing values
    df.dropna(axis=0, subset= ["diag_1", "diag_2", "diag_3"], inplace = True)

    #remove patients that expired or went to hospice
    df=df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    #removal of columns with many missing values and columns that are not useful for prediction
    df = df.drop(columns = [
        "weight",
        "encounter_id",
        "patient_nbr",
        "max_glu_serum",
        "A1Cresult",
        "medical_specialty",
        "payer_code"])

    #classify the target variable into binary classes
    df["readmitted"] = df["readmitted"].replace({"NO": 0, ">30": 0, "<30": 1})
    df["readmitted"] =df["readmitted"].astype(int)

    #filling missing values with unknown for the race column
    df["race"] = df["race"].fillna("Unknown")


    #dataset spliting into features and target variable
    Y = df["readmitted"]
    X = df.drop(columns = ["readmitted"])

    X['diag_1'] = X['diag_1'].apply(map_icd9)
    X['diag_2'] = X['diag_2'].apply(map_icd9)
    X['diag_3'] = X['diag_3'].apply(map_icd9)

    # Dataset spliting into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y, 
        random_state = 18,
        test_size = 0.25,
        stratify = Y)
    
    
    # Creation of OrdinalEncoder
    ode = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)

    # Categorical columns
    cat_cols = X_train.select_dtypes(include = ['object', 'str']).columns

    # Encoding in train and test set
    X_train[cat_cols] = ode.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = ode.transform(X_test[cat_cols])


    # Save the encoder in a file
    with open('encoder.pkl','wb') as file:
        pickle.dump(ode, file)



    
    return X_train, X_test, Y_train, Y_test, ode
    


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, ode = data_preprocessing()
    
