NAS Data Analysis Dashboard
This is a Streamlit-based dashboard for analyzing NAS (National Health/Patient) datasets. It allows users to upload one or multiple CSV/XLSX files and provides interactive visualizations for diagnoses, medications, images, referrals, and demographic stratifications.

Features
Upload NAS Files

Supports multiple CSV or Excel files.
Automatically merges uploaded files.
Diagnosis Classification

Classifies diagnoses as Single or Multiple based on text patterns.
Top Diagnoses Visualization

Displays top 20 diagnoses overall.
Separately shows top 20 single and multiple diagnoses.
Stratifies by patients with or without images.
Images Analysis

Shows patients with vs without images.
Medications

Splits and counts prescribed medicines.
Displays top 20 and complete list of medicines.
Referrals

Counts suggested referrals by facility (PHC, DH, SDH).
Visit Date Range

Displays date range of visits and total visit count.
Demographic Stratification

Gender distribution by month.
Age group distribution by month.
Installation
Clone this repository: git clone https://github.com/KruthikaTS/Ai_intern.git

Create a virtual environment: python -m venv venv On Windows: venv\Scripts\activate

Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app_streamlit.py
