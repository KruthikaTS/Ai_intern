# NAS Data Analysis Dashboard

This is a Streamlit-based dashboard for analyzing Healthcare Datasets. It allows users to upload one or multiple CSV/XLSX files and provides interactive visualizations for diagnoses, medications, images, referrals, and demographic stratifications.

---

## **Features**

1. **Upload Healthcare Files**  
   - Supports multiple CSV or Excel files.  
   - Automatically merges uploaded files.  

2. **Diagnosis Classification**  
   - Classifies diagnoses as **Single** or **Multiple** based on text patterns.  

3. **Top Diagnoses Visualization**  
   - Displays top 20 diagnoses overall.  
   - Separately shows top 20 single and multiple diagnoses.  
   - Stratifies by patients with or without images.  

4. **Images Analysis**  
   - Shows patients with vs without images.  

5. **Medications**  
   - Splits and counts prescribed medicines.  
   - Displays top 20 and complete list of medicines.  

6. **Referrals**  
   - Counts suggested referrals by facility (PHC, DH, SDH).  

7. **Visit Date Range**  
   - Displays date range of visits and total visit count.  

8. **Demographic Stratification**  
   - Gender distribution by month.  
   - Age group distribution by month.  

---

## **Installation**
1. **Clone this repository:**
```bash
git clone https://github.com/KruthikaTS/Ai_intern.git
cd Ai_intern

2. *Create a virtual environment:*
python -m venv venv
On Windows: venv\Scripts\activate

3. *Install dependencies:*
pip install -r requirements.txt

4. *Run the Streamlit app:*
streamlit run app_streamlit.py




