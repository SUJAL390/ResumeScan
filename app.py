import os
import pickle
import re
import nltk
import streamlit as st
from PyPDF2 import PdfReader

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Ensure the script uses the correct working directory
script_dir = os.path.dirname(__file__)
tfidf_path = os.path.join(script_dir, 'tfidf.pkl')
clf_path = os.path.join(script_dir, 'clf.pkl')

# Load the models
with open(tfidf_path, 'rb') as f:
    tfidfd = pickle.load(f)

with open(clf_path, 'rb') as f:
    clf = pickle.load(f)

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle cases where text extraction might fail
    return text

# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                resume_text = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = uploaded_file.read().decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
            26:"React Developer",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
