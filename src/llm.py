# Import packages
from dotenv import load_dotenv
import os
import google.generativeai as genai


# Load environment variables from the .env file (API key)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Authentication with Google using API key
genai.configure(api_key=api_key)

def generate_summary(data, predicted_probability, predicted_class):

    # Write the prompt
    prompt = f"""
You are a medical assistant who is trying to explain the predicted risk of hospital readmission for diabetes patients.
Give as output a simple explanation to the medical stuff for the prediction with no medical terms in 2-3 sentences based on the following:
The posibility of the patient's readmission is {predicted_probability*100} and the prediction is {predicted_class} (1 means readmission in less than 30 days while 0 means no readmission).
The patient's most important data that you can use for the prediction's explanation are:
age: {data['age']}
time in hospital in the last hospitalization (in days): {data['time_in_hospital']}
number of inpatients in the past: {data['number_inpatient']}
number of diagnoses in total: {data['number_diagnoses']}
number of medications in the last hospitalization: {data['num_medications']}
the main diagnosis (ICD-9): {data['diag_1']}

You can judge some of this data to justify the prediction. The main diagnosis is an ICD-9 code, so if you use it translate it into its clinical meaning.
"""

    # Choose which model
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    return response.text