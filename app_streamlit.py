import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import unicodedata
from collections import Counter
from rapidfuzz import process, fuzz




import spacy
nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
st.title("NAS Data Analysis Dashboard")

# ------------------------
# 1. Upload CSV/Excel (single or multiple)
# ------------------------
uploaded_files = st.file_uploader(
    "Upload NAS CSV/XLSX file(s)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key="nas_file_uploader"
)

if uploaded_files:
    df_list = []

    for file in uploaded_files:
        if file.name.endswith('.csv'):
            temp_df = pd.read_csv(file)
        else:
            temp_df = pd.read_excel(file)
        st.write(f"{file.name} uploaded: {len(temp_df)} rows")
        df_list.append(temp_df)

    # Merge all uploaded files
    df = pd.concat(df_list, ignore_index=True)
    st.success(f"All files merged! Total rows: {len(df)}")

    # ------------------------
    # 2. Diagnosis Classification
    # ------------------------
    def classify_diagnosis(diagnosis):
        if pd.isna(diagnosis):
            return 'Unknown'
        diag = diagnosis.strip()
        if ',' in diag:
            after_comma = diag.split(',', 1)[1].strip()
            if len(after_comma) > 5:
                return 'Multiple'
            else:
                return 'Single'
        if re.search(r'[a-z][A-Z]', diag):
            return 'Multiple'
        return 'Single'

    df['Diagnosis_type'] = df['Primary & Provisional'].apply(classify_diagnosis)

    # Separate single and multiple
    single_diag_df = df[df['Diagnosis_type'] == 'Single']
    multiple_diag_df = df[df['Diagnosis_type'] == 'Multiple']

    # ------------------------
    # 3. Top 20 Diagnoses
    # ------------------------
    def add_labels(ax):
        """Add data labels to a horizontal or vertical bar chart"""
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

    st.subheader("Top 20 Diagnoses")
    diagnosis_counts = df['Primary & Provisional'].value_counts()
    fig, ax = plt.subplots(figsize=(10,6))
    bars = diagnosis_counts.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Diagnosis")
    ax.set_title("Top 20 Diagnoses in NAS")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 4. Single vs Multiple Diagnoses Top 20
    # ------------------------
    st.subheader("Top 20 Single Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = single_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("Top 20 Multiple Diagnoses")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = multiple_diag_df['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 5. Images Stats
    # ------------------------
    df['Has_image'] = df['Images'].str.contains('http', na=False)
    image_counts = df['Has_image'].value_counts()
    total_rows = len(df)

    st.subheader("Patients with vs without Images")
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(image_counts.index.map({True: 'Has Image', False: 'No Image'}), image_counts.values, color='red')
    ax.set_xlabel("Image Availability")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Patients with vs without Images (Total Rows={total_rows})")
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 6. Single/Multiple for sub-samples (with and without images)
    # ------------------------
    st.subheader("Top 20 Single/Multiple Diagnoses - Patients with Images")
    df_with_images = df[df['Has_image']==True]
    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_with_images[df_with_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_with_images[df_with_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Multiple Diagnosis")
    ax.set_title("Top 20 Multiple Diagnoses - Patients with Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("Top 20 Single/Multiple Diagnoses - Patients without Images")
    df_no_images = df[df['Has_image']==False]
    fig, ax = plt.subplots(figsize=(10,6))
    bars = df_no_images[df_no_images['Diagnosis_type']=='Single']['Primary & Provisional'].value_counts().head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Single Diagnosis")
    ax.set_title("Top 20 Single Diagnoses - Patients without Images")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    multiple_no_images = df_no_images[df_no_images['Diagnosis_type']=='Multiple']['Primary & Provisional'].value_counts().head(20)
    if not multiple_no_images.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        bars = multiple_no_images.plot(kind='barh', color='red', ax=ax)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Multiple Diagnosis")
        ax.set_title("Top 20 Multiple Diagnoses - Patients without Images")
        plt.gca().invert_yaxis()
        add_labels(ax)
        st.pyplot(fig)
    else:
        st.write("No patients without images have multiple diagnoses in this dataset.")

    # ------------------------
    # 7. Medications
    # ------------------------
    def split_medicines(row):
        if pd.isna(row):
            return []
        parts = re.split(r',(?![^()]*\))', row)
        return [p.strip().rstrip('.') for p in parts if p.strip()]

    df['Medicines_split'] = df['Medicines'].apply(split_medicines)
    all_meds = df['Medicines_split'].explode().reset_index(drop=True)
    med_freq = all_meds.value_counts()

    st.subheader("Top 20 Prescribed Medicines")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = med_freq.head(20).plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("Top 20 Prescribed Medicines")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    st.subheader("All Prescribed Medicines")
    fig, ax = plt.subplots(figsize=(12,10))
    bars = med_freq.plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Medicine")
    ax.set_title("All Prescribed Medicines Frequency")
    plt.gca().invert_yaxis()
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 8. Referrals
    # ------------------------
    phc_count = df['Referral_advice'].str.contains('PHC', na=False).sum()
    dh_count = df['Referral_advice'].str.contains('DH', na=False).sum()
    sdh_count = df['Referral_advice'].str.contains('SDH', na=False).sum()

    referral_counts = pd.DataFrame({
        'Facility': ['PHC', 'DH', 'SDH'],
        'Count': [phc_count, dh_count, sdh_count]
    })

    st.subheader("Referrals Suggested by Facility")
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(referral_counts['Facility'], referral_counts['Count'], color='red')
    ax.set_xlabel("Facility")
    ax.set_ylabel("Number of Referrals")
    ax.set_title("Referrals Suggested by Facility")
    add_labels(ax)
    st.pyplot(fig)

    # ------------------------
    # 9.Visit Date Range
    # ------------------------
    df['Visit_started_date'] = pd.to_datetime(df['Visit_started_date'], errors='coerce')
    dates = df['Visit_started_date'].dropna()
    start_date = dates.min()
    end_date = dates.max()
    total_days = (end_date - start_date).days + 1
    total_visits = len(dates)

    st.write(f"**Visit date range:** {start_date.date()} to {end_date.date()} ({total_days} days)")
    st.write(f"**Total visits:** {total_visits}")

    # ------------------------
    # Month Extraction
    # ------------------------
    df['Month'] = df['Visit_started_date'].dt.to_period('M')
    df['Month_Label'] = df['Visit_started_date'].dt.strftime('%Y-%m (%b)')

    # ------------------------
    # 10. Gender Stratification
    # ------------------------
    st.subheader("Gender Stratification by Month")
    df['Gender_Label'] = df['Gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Other')
    gender_month = df.groupby(['Month_Label','Gender_Label'], observed=False).size().unstack(fill_value=0)
    st.dataframe(gender_month)

    gender_colors = {'Male': 'blue', 'Female': 'pink', 'Other': 'purple'}
    ax = gender_month.plot(
        kind='bar',
        figsize=(10,6),
        color=[gender_colors.get(col, 'gray') for col in gender_month.columns]
    )
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Gender Stratification of Patients by Month")
    plt.xticks(rotation=45)
    plt.legend(title="Gender")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(plt.gcf())

    # ------------------------
    # 11. Age Stratification
    # ------------------------
    st.subheader("Age Stratification by Month")
    bins = [0, 12, 18, 59, 200]
    labels = ['Pediatric (0-12)', 'Adolescent (13-18)', 'Adults (19-59)', 'Elderly (60+)']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    age_month = df.groupby(['Month_Label','Age_Group'], observed=False).size().unstack(fill_value=0)
    st.dataframe(age_month)

    age_colors = {
        'Pediatric (0-12)': 'yellow',
        'Adolescent (13-18)': 'orange',
        'Adults (19-59)': 'teal',
        'Elderly (60+)': 'purple'
    }

    ax = age_month.plot(
        kind='bar',
        stacked=True,
        figsize=(12,6),
        color=[age_colors.get(col, 'gray') for col in age_month.columns]
    )
    plt.xlabel("Month")
    plt.ylabel("Number of Patients")
    plt.title("Age Group Stratification by Month")
    plt.xticks(rotation=45)
    plt.legend(title="Age Group")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(plt.gcf())

    # ------------------------
    # 12.Full Symptoms and Associated Symptoms Visualization
    # ------------------------
    st.subheader("Symptoms Frequency Analysis")
    # ------------------------
    # 1. Known truncation fixes
    fix_map = {
        'Coug': 'Cough',
        'Joi': 'Joint pain',
        'Burning feeling in t': 'Burning feeling in a throat at night/early in the morning',
        'Pain/Tightness of ch': 'Pain/Tightness of chest'
    }

    # ------------------------
    # 2. Merge special multi-part symptoms
    def merge_leg_knee(symptoms_list):
        merged = []
        skip_next = False
        for i in range(len(symptoms_list)):
            if skip_next:
                skip_next = False
                continue
            s = symptoms_list[i].strip()
            # Merge 'Leg' + 'Knee or Hip Pain'
            if s.lower() == 'leg' and i+1 < len(symptoms_list) and 'knee or hip pain' in symptoms_list[i+1].lower():
                merged.append('Leg, Knee or Hip Pain')
                skip_next = True
            else:
                merged.append(s)
        return merged

    # ------------------------
    # 3. Function to extract and clean individual symptoms
    def extract_actual_symptoms(row):
        if pd.isna(row) or str(row).strip().lower() == 'null':
            return []

        s = str(row)
        s = s.replace('\n', ' ')  # remove line breaks
        s = ''.join(ch for ch in unicodedata.normalize('NFKC', s) if ch.isprintable())  # remove invisible chars

        # Split on commas
        parts = [p.strip() for p in s.split(',') if p.strip()]
        
        # Merge 'Leg' + 'Knee or Hip Pain'
        parts = merge_leg_knee(parts)

        symptoms = []
        for part in parts:
            # Remove everything after ►
            symptom = part.split('►')[0].strip()

            # Remove trailing descriptors if likely timing/description
            if '-' in symptom and len(symptom.split('-')[-1].strip()) > 15:
                symptom = '-'.join(symptom.split('-')[:-1]).strip()

            # Apply known truncation mapping
            symptom = fix_map.get(symptom, symptom)
            if len(symptom) > 2:
                symptoms.append(symptom)
        return symptoms

    # ------------------------
    # 4. Function to normalize and clean exploded symptoms
    def clean_symptoms(column_data):
        all_symptoms = column_data.apply(extract_actual_symptoms).explode().reset_index(drop=True)
        all_symptoms = all_symptoms.str.strip()
        all_symptoms = all_symptoms[all_symptoms != '']

        # Remove placeholders / invalid entries
        remove_words = ['Other', 'Misc', 'NA', 'None']
        all_symptoms = all_symptoms[~all_symptoms.isin(remove_words)]

        # Mostly ASCII
        def is_mostly_ascii(s, threshold=0.9):
            if not s or not isinstance(s, str):
                return False
            ascii_chars = sum(1 for c in s if ord(c) < 128)
            return ascii_chars / len(s) >= threshold
        all_symptoms = all_symptoms[all_symptoms.apply(is_mostly_ascii)]

        # Remove numbers, months, very short, or timing descriptors
        def is_valid_symptom(s):
            s_clean = s.strip().lower()
            if any(x in s_clean for x in ['month', 'months', 'yr', 'yrs']):
                return False
            if s_clean.isnumeric():
                return False
            if len(s_clean) < 3:
                return False
            if s_clean.startswith('timing') or s_clean.startswith('lasting'):
                return False
            return True
        all_symptoms = all_symptoms[all_symptoms.apply(is_valid_symptom)]

        # Remove trailing punctuation and normalize
        all_symptoms = all_symptoms.str.rstrip('. ,;:').str.strip().str.lower()

        # Merge truncated symptoms
        reference_symptoms = all_symptoms.unique().tolist()
        def merge_truncated_symptoms(s, reference_list):
            s_clean = s.strip()
            for ref in reference_list:
                if ref.startswith(s_clean) or s_clean in ref:
                    return ref
            return s_clean
        all_symptoms = all_symptoms.apply(lambda x: merge_truncated_symptoms(x, reference_symptoms))

        return all_symptoms

    # ------------------------
    # 5. Process Symptoms column
    symptoms_cleaned = clean_symptoms(df['Symptoms'])
    symp_counts = symptoms_cleaned.value_counts()

    # Top 20 Symptoms
    st.subheader("Symptoms Frequency Analysis (Top 20)")
    top_n = 20
    top_symp = symp_counts.head(top_n)
    fig_height = max(6, len(top_symp) * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    top_symp.plot(kind='barh', color='blue', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Symptom")
    ax.set_title("Top 20 Symptoms")
    plt.gca().invert_yaxis()
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    st.pyplot(fig)

    # All Symptoms
    st.subheader("Symptoms Frequency Analysis (All)")
    fig_height_all = max(8, len(symp_counts) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height_all))
    symp_counts.plot(kind='barh', color='blue', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Symptom")
    ax.set_title("Symptoms Frequency (All)")
    plt.gca().invert_yaxis()
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    st.pyplot(fig)

    # ------------------------
    # 6. Process Associated_Symptoms column
    assoc_symptoms_cleaned = clean_symptoms(df['Associated_Symptoms'])
    assoc_counts = assoc_symptoms_cleaned.value_counts()

    # Top 20 Associated Symptoms
    st.subheader("Associated Symptoms Frequency Analysis (Top 20)")
    top_assoc = assoc_counts.head(top_n)
    fig_height = max(6, len(top_assoc) * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    top_assoc.plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Associated Symptom")
    ax.set_title("Top 20 Associated Symptoms")
    plt.gca().invert_yaxis()
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    st.pyplot(fig)

    # All Associated Symptoms
    st.subheader("Associated Symptoms Frequency Analysis (All)")
    fig_height_all = max(8, len(assoc_counts) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height_all))
    assoc_counts.plot(kind='barh', color='red', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Associated Symptom")
    ax.set_title("Associated Symptoms Frequency (All)")
    plt.gca().invert_yaxis()
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    st.pyplot(fig)


    # ------------------------
# 2. Chief Complaint → Associated Symptoms Mapping (Excel-based)
# ------------------------
    st.subheader("Chief Complaint → Associated Symptoms Mapping")

    if "Chief_complaint" in df.columns and "Associated_Symptoms" in df.columns:
        
        mapping_rows = []

        for _, row in df.iterrows():
            chief_text = str(row["Chief_complaint"])
            assoc_text = str(row["Associated_Symptoms"]).strip()

            # Extract all chief complaints in the cell (ignore HTML tags)
            chief_matches = re.findall(r'►<b>(.*?)</b>:', chief_text)
            chief_complaints = [c.strip() for c in chief_matches if not re.search(r'associated symptoms', c, re.I)]

            if not chief_complaints:
                # If regex didn't match, fallback to entire cell
                chief_complaints = [chief_text.strip()]

            # Map each chief complaint to the associated symptoms column value
            for chief in chief_complaints:
                mapping_rows.append({
                    "Chief_Complaint": chief,
                    "Associated_Symptoms": assoc_text
                })

        # Build DataFrame
        mapping_df = pd.DataFrame(mapping_rows).drop_duplicates().reset_index(drop=True)
        
        # Show preview
        st.write("✅ Extracted mapping preview (first 20 rows):")
        st.dataframe(mapping_df.head(20), use_container_width=True)
        
        # Download CSV button
        csv_data = mapping_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Chief Complaint ↔ Associated Symptoms Mapping CSV",
            data=csv_data,
            file_name="chief_complaint_associated_symptoms_mapping.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Columns 'Chief_complaint' and/or 'Associated_Symptoms' not found in uploaded dataset.")

    # ------------------------

#nlp


        st.write("Step 1: Clinical notes loaded")


        # 1. Extract chief complaint, reported, denied symptoms
        # ------------------------
        def extract_clinical_info(text):
            text = str(text)
            
            # Chief complaint
            chief_match = re.search(r'Chief_complaint:\s*►\*\*(.*?)\*\*', text, re.IGNORECASE)
            chief_complaint = chief_match.group(1).strip() if chief_match else "Unknown"

            # Reported symptoms
            reported_match = re.search(r'Patient reports\s*-\s*(.*?)(?:\n•|Patient denies|$)', text, re.DOTALL | re.IGNORECASE)
            reported = []
            if reported_match:
                reported_text = reported_match.group(1).strip()
                reported = [line.strip() for line in re.split(r'\n|•', reported_text) if line.strip()]
            
            # Denied symptoms
            denied_match = re.search(r'Patient denies\s*-\s*(.*?)(?:\n•|$)', text, re.DOTALL | re.IGNORECASE)
            denied = []
            if denied_match:
                denied_text = denied_match.group(1).strip()
                denied = [line.strip() for line in re.split(r',|•|\n', denied_text) if line.strip()]
            
            return chief_complaint, reported, denied

        df[['Chief_complaint_extracted', 'Reported_symptoms_raw', 'Denied_symptoms_raw']] = df['Clinical_notes'].apply(
            lambda x: pd.Series(extract_clinical_info(x))
        )

        # ------------------------
        # 2. NLP normalization
        # ------------------------
        def normalize_list(symptoms_list):
            normalized = []
            for s in symptoms_list:
                doc = nlp(s.lower())
                tokens = [token.lemma_ for token in doc if token.is_alpha]
                if tokens:
                    normalized.append(" ".join(tokens))
            return normalized

        df['Reported_symptoms_norm'] = df['Reported_symptoms_raw'].apply(normalize_list)
        df['Denied_symptoms_norm'] = df['Denied_symptoms_raw'].apply(normalize_list)

        st.write("Step 2: Extracting chief, reported, denied symptoms")
       
        # ------------------------
        # 3. Semantic clustering to merge similar symptoms
        # ------------------------
        def cluster_symptoms(symptom_series, similarity_threshold=0.85):
            all_symptoms = symptom_series.explode().dropna().unique().tolist()
            clusters = []
            for s in all_symptoms:
                s_vec = nlp(s).vector.reshape(1, -1)
                found_cluster = False
                for cluster in clusters:
                    rep_vec = nlp(cluster[0]).vector.reshape(1, -1)
                    sim = cosine_similarity(s_vec, rep_vec)[0][0]
                    if sim >= similarity_threshold:
                        cluster.append(s)
                        found_cluster = True
                        break
                if not found_cluster:
                    clusters.append([s])
                    
            
            # Build mapping from symptom -> cluster representative
            mapping = {}
            for cluster in clusters:
                rep = cluster[0]  # first symptom in cluster as representative
                for s in cluster:
                    mapping[s] = rep
            return mapping

        reported_mapping = cluster_symptoms(df['Reported_symptoms_norm'])
        denied_mapping = cluster_symptoms(df['Denied_symptoms_norm'])

        # Apply mapping to normalized lists
        def apply_mapping(lst, mapping):
            return [mapping.get(s, s) for s in lst]

        df['Reported_symptoms_final'] = df['Reported_symptoms_norm'].apply(lambda x: apply_mapping(x, reported_mapping))
        df['Denied_symptoms_final'] = df['Denied_symptoms_norm'].apply(lambda x: apply_mapping(x, denied_mapping))
        st.write("Step 3: Normalizing symptoms")
        # ------------------------
        # 4. Count frequencies
        # ------------------------
        reported_all = df['Reported_symptoms_final'].explode().dropna()
        denied_all = df['Denied_symptoms_final'].explode().dropna()

        reported_counts = reported_all.value_counts()
        denied_counts = denied_all.value_counts()
        st.write("Step 4: Clustering similar symptoms")

        # ------------------------
        # 5. Plot function
        # ------------------------
        def plot_symptoms(counts, color, title, top_n=20):
            top_counts = counts.head(top_n)
            fig_height = max(6, len(top_counts)*0.5)
            fig, ax = plt.subplots(figsize=(12, fig_height))
            top_counts.plot(kind='barh', color=color, ax=ax)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Symptom")
            ax.set_title(title)
            plt.gca().invert_yaxis()
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
            st.pyplot(fig)

        # ------------------------
        # 6. Bar charts
        # ------------------------
        st.subheader("Reported Symptoms (Top 20 - Blue)")
        plot_symptoms(reported_counts, 'blue', "Top 20 Reported Symptoms", top_n=20)

        st.subheader("Reported Symptoms (All - Blue)")
        plot_symptoms(reported_counts, 'blue', "All Reported Symptoms", top_n=len(reported_counts))

        st.subheader("Denied Symptoms (Top 20 - Red)")
        plot_symptoms(denied_counts, 'red', "Top 20 Denied Symptoms", top_n=20)

        st.subheader("Denied Symptoms (All - Red)")
        plot_symptoms(denied_counts, 'red', "All Denied Symptoms", top_n=len(denied_counts))

        # ------------------------
        # 7. Table: Chief Complaint | Reported | Denied
        # ------------------------
        st.subheader("Chief Complaint → Reported & Denied Symptoms Table")
        table_df = df[['Chief_complaint_extracted', 'Reported_symptoms_final', 'Denied_symptoms_final']].copy()
        table_df['Reported_symptoms_final'] = table_df['Reported_symptoms_final'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        table_df['Denied_symptoms_final'] = table_df['Denied_symptoms_final'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        st.dataframe(table_df, use_container_width=True)


#new refined code but doesxnt show op


        # 2. Function to extract chief complaint, reported, denied symptoms
        # ------------------------
    def extract_clinical_info(text):
        text = str(text)
            
            # Chief complaint
        chief_match = re.search(r'Chief_complaint:\s*►\*\*(.*?)\*\*', text, re.IGNORECASE)
        chief_complaint = chief_match.group(1).strip() if chief_match else "Unknown"

            # Reported symptoms
        reported_match = re.search(r'Patient reports\s*-\s*(.*?)(?:\n•|Patient denies|$)', text, re.DOTALL | re.IGNORECASE)
        reported = []
        if reported_match:
            reported_text = reported_match.group(1).strip()
            reported = [line.strip() for line in re.split(r'\n|•', reported_text) if line.strip()]
            
            # Denied symptoms
        denied_match = re.search(r'Patient denies\s*-\s*(.*?)(?:\n•|$)', text, re.DOTALL | re.IGNORECASE)
        denied = []
        if denied_match:
            denied_text = denied_match.group(1).strip()
            denied = [line.strip() for line in re.split(r',|•|\n', denied_text) if line.strip()]
            
            return chief_complaint, reported, denied

        # ------------------------
        # 3. Loop over rows (so Streamlit shows progress)
        # ------------------------
        chiefs, reported_raw, denied_raw = [], [], []
        for i, note in enumerate(df['Clinical_notes']):
            chief, reported, denied = extract_clinical_info(note)
            chiefs.append(chief)
            reported_raw.append(reported)
            denied_raw.append(denied)
            
            if i % 50 == 0:
                st.write(f"Processed {i}/{len(df)} rows")

        df['Chief_complaint_extracted'] = chiefs
        df['Reported_symptoms_raw'] = reported_raw
        df['Denied_symptoms_raw'] = denied_raw

        st.write("Step 2: Extraction done!")

        # ------------------------
        # 4. NLP normalization (lemmatize & lowercase)
        # ------------------------
        def normalize_list(symptoms_list):
            normalized = []
            for s in symptoms_list:
                doc = nlp(s.lower())
                tokens = [token.lemma_ for token in doc if token.is_alpha]
                if tokens:
                    normalized.append(" ".join(tokens))
            return normalized

        df['Reported_symptoms_norm'] = df['Reported_symptoms_raw'].apply(normalize_list)
        df['Denied_symptoms_norm'] = df['Denied_symptoms_raw'].apply(normalize_list)

        # ------------------------
        # 5. Count frequencies
        # ------------------------
        reported_all = df['Reported_symptoms_norm'].explode().dropna()
        denied_all = df['Denied_symptoms_norm'].explode().dropna()

        reported_counts = reported_all.value_counts()
        denied_counts = denied_all.value_counts()

        # ------------------------
        # 6. Plot function
        # ------------------------
        def plot_symptoms(counts, color, title, top_n=20):
            top_counts = counts.head(top_n)
            fig_height = max(6, len(top_counts)*0.5)
            fig, ax = plt.subplots(figsize=(12, fig_height))
            top_counts.plot(kind='barh', color=color, ax=ax)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Symptom")
            ax.set_title(title)
            plt.gca().invert_yaxis()
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
            st.pyplot(fig)

        # ------------------------
        # 7. Bar charts
        # ------------------------
        st.subheader("Reported Symptoms (Top 20 - Blue)")
        plot_symptoms(reported_counts, 'blue', "Top 20 Reported Symptoms", top_n=20)

        st.subheader("Reported Symptoms (All - Blue)")
        plot_symptoms(reported_counts, 'blue', "All Reported Symptoms", top_n=len(reported_counts))

        st.subheader("Denied Symptoms (Top 20 - Red)")
        plot_symptoms(denied_counts, 'red', "Top 20 Denied Symptoms", top_n=20)

        st.subheader("Denied Symptoms (All - Red)")
        plot_symptoms(denied_counts, 'red', "All Denied Symptoms", top_n=len(denied_counts))

        # ------------------------
        # 8. Table: Chief Complaint | Reported | Denied
        # ------------------------
        st.subheader("Chief Complaint → Reported & Denied Symptoms Table")
        table_df = df[['Chief_complaint_extracted', 'Reported_symptoms_norm', 'Denied_symptoms_norm']].copy()
        table_df['Reported_symptoms_norm'] = table_df['Reported_symptoms_norm'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        table_df['Denied_symptoms_norm'] = table_df['Denied_symptoms_norm'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        st.dataframe(table_df, use_container_width=True)
