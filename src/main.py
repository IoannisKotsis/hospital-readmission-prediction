from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi import FastAPI
from preprocessing import map_icd9
from database import insert_prediction
from llm import generate_summary


# Loading model and encoder
with open('../src/model.pkl','rb') as file:
    model = pickle.load(file)

with open('../src/encoder.pkl','rb') as file:
    encoder = pickle.load(file)



# API building
app = FastAPI()

class PatientData(BaseModel):
    race : str
    gender : str
    age : str
    admission_type_id : int
    discharge_disposition_id : int
    admission_source_id : int
    time_in_hospital : int
    num_lab_procedures : int
    num_procedures : int
    num_medications : int
    number_outpatient : int
    number_emergency : int
    number_inpatient : int
    diag_1 : str
    diag_2 : str
    diag_3 : str
    number_diagnoses : int
    metformin : str
    repaglinide : str
    nateglinide : str
    chlorpropamide : str
    glimepiride : str
    acetohexamide : str
    glipizide : str
    glyburide : str
    tolbutamide : str
    pioglitazone : str
    rosiglitazone : str
    acarbose : str
    miglitol : str
    troglitazone : str
    tolazamide : str
    examide : str
    citoglipton : str
    insulin : str
    glyburide_metformin : str
    glipizide_metformin : str
    glimepiride_pioglitazone : str
    metformin_rosiglitazone : str
    metformin_pioglitazone : str
    med_change : str
    diabetes_med : str


# API request
@app.post("/predict")
def predict(patient: PatientData):

    # Raw data -> dict -> dataframe
    raw_data = patient.model_dump()
    patient_df = pd.DataFrame([raw_data])

    # icd9 mapping
    patient_df['diag_1'] = patient_df['diag_1'].apply(map_icd9)
    patient_df['diag_2'] = patient_df['diag_2'].apply(map_icd9)
    patient_df['diag_3'] = patient_df['diag_3'].apply(map_icd9)

    # Encoding
    cat_cols = encoder.feature_names_in_
    patient_df[cat_cols] = encoder.transform(patient_df[cat_cols])

    # Model prediction
    predicted_probability = float(model.predict_proba(patient_df)[:,1][0])
    predicted_class = int(model.predict(patient_df)[0])
    model_version = "v1.0"

    # In case LLM crashes or cannot generate anymore
    try:
        summary = generate_summary(raw_data, predicted_probability, predicted_class)
    except Exception as e:
        print(f"LLM summary failed: {e}")
        summary = None

    # Insert prediction to database
    raw_data = {
        **raw_data,
        "predicted_probability": predicted_probability,
        "predicted_class": predicted_class,
        "model_version": model_version,
        "llm_summary" : summary
        }
    
    insert_prediction(raw_data)

    return{
        "predicted_probability" : predicted_probability,
        "predicted_class" : predicted_class,
        "summary" : summary
    }

