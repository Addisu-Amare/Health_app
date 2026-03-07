import streamlit as st
import pickle
import pandas as pd
import numpy as np
from thefuzz import process
import ast

# Page configuration with Amharic title
st.set_page_config(
    page_title="የጤና እንክብካቤ ማዕከል | Health Care Hub", 
    page_icon="🩺", 
    layout='wide'
)

# Custom CSS for Amharic font support
st.markdown("""
<style>
    /* Import Amharic font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;500;600;700&display=swap');
    
    /* Apply Amharic font to all text */
    * {
        font-family: 'Noto Sans Ethiopic', 'Poppins', sans-serif;
    }
    
    /* Amharic title styling */
    .amharic-title {
        font-size: 48px;
        font-weight: 700;
        color: #1e90ff;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 0px 0px 10px rgba(30, 144, 255, 0.3);
    }
    
    .amharic-subtitle {
        font-size: 20px;
        color: #bdbdbd;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    .amharic-text {
        font-size: 16px;
        line-height: 1.8;
    }
    
    /* Bilingual headers */
    .bilingual-header {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 15px;
        border-left: 5px solid #1e90ff;
        padding-left: 15px;
    }
    
    .bilingual-header small {
        font-size: 16px;
        color: #bdbdbd;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Main Title - Amharic and English
st.markdown("""
<div style='text-align: center;'>
    <h1 class='amharic-title'>የጤና እንክብካቤ ማዕከል</h1>
    <h2 style='color: #1e90ff; margin-top: -10px;'>Health Care Hub</h2>
    <p class='amharic-subtitle'>በአይ የሚመራ የበሽታ መመርመሪያ እና የሕክምና ምክር ሥርዓት</p>
    <p class='amharic-subtitle'>AI-Powered Disease Prediction & Medical Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with Amharic description
with st.sidebar:
    st.markdown("""
    <h2 style='color: #ffffff; text-align: center;'>
        📌 መግለጫ<br>
        <small style='font-size: 16px;'>Description</small>
    </h2>
    """, unsafe_allow_html=True)
    
    st.image("utils/ph3.jpg", use_container_width=True)
    
    st.markdown("""
    <div style='background: rgba(30, 144, 255, 0.1); padding: 15px; border-radius: 10px;'>
        <p style='color: #ffffff; line-height: 1.8;'>
            <strong>🇪🇹 አማርኛ፦</strong><br>
            የበሽታ መመርመሪያ እና የሕክምና ምክር ሥርዓቱ የምልክቶችን በመተንተን፣ በሽታዎችን ለመተንበይ፣ የጤና ስጋቶችን ለመገምገም እና የግል ሕክምናዎችን ለመጠቆም አርቴፊሻል ኢንተሊጀንስ ይጠቀማል።
        </p>
        <p style='color: #bdbdbd; line-height: 1.6;'>
            <strong>🇬🇧 English:</strong><br>
            The Disease Prediction & Medical Recommendation system uses AI to analyze symptoms, predict diseases, assess health risks, and suggest personalized treatments—enhancing early diagnosis and improving healthcare decisions for better patient outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    try:
        sym_des = pd.read_csv("data/Disease-Prediction-and-Medical dataset/symptoms_df.csv")
        precautions = pd.read_csv("data/Disease-Prediction-and-Medical dataset/precautions_df.csv")
        workout = pd.read_csv("data/Disease-Prediction-and-Medical dataset/workout_df.csv")
        description = pd.read_csv("data/Disease-Prediction-and-Medical dataset/description.csv")
        medications = pd.read_csv("data/Disease-Prediction-and-Medical dataset/medications.csv")
        diets = pd.read_csv("data/Disease-Prediction-and-Medical dataset/diets.csv")
        model = pickle.load(open('models/first_feature_models/RandomForest.pkl', 'rb'))
        return sym_des, precautions, workout, description, medications, diets, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

sym_des, precautions, workout, description, medications, diets, model = load_data()

disease_names = list(description['Disease'].unique()) if description is not None else []

symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

def information(predicted_dis):
    try:
        disease_desciption = description.loc[description['Disease'] == predicted_dis, 'Description'].values[0]
        disease_precautions = precautions.loc[precautions['Disease'] == predicted_dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
        disease_medications = ast.literal_eval(medications.loc[medications['Disease'] == predicted_dis, 'Medication'].values[0])
        disease_diet = ast.literal_eval(diets.loc[diets['Disease'] == predicted_dis, 'Diet'].values[0])
        disease_workout = workout.loc[workout['disease'] == predicted_dis, 'workout'].values.tolist()
        return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout
    except Exception:
        return "Description not available", [], [], [], []

def predicted_value(patient_symptoms):
    try:
        i_vector = np.zeros(len(symptoms_list_processed))
        for symptom in patient_symptoms:
            i_vector[symptoms_list_processed[symptom]] = 1
        return diseases_list.get(model.predict([i_vector])[0], "Unknown Disease")
    except Exception:
        return "Prediction Error"

def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    return closest_match if score >= 80 else None

# Disease Prediction Section - Bilingual
st.markdown("""
<div class='bilingual-header'>
    🩺 በምልክቶች ላይ የተመሠረተ የበሽታ መመርመሪያ <br>
    <small>Disease Prediction Based on Symptoms</small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background: rgba(30, 144, 255, 0.05); padding: 10px; border-radius: 8px; margin-bottom: 15px;'>
    <p>🇪🇹 <em>በጣም ትክክለኛ ውጤት ለማግኘት በተቻለዎት መጠን ብዙ ምልክቶችን ያስገቡ።</em></p>
    <p>🇬🇧 <em>To get the best and most accurate results, provide as many symptoms as possible.</em></p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "🇪🇹 ምልክቶችን ያስገቡ (በነጠላ ሰረዝ ተለያይተው) / Enter symptoms (comma-separated):", 
    placeholder="ለምሳሌ / e.g., headache, constipation, nausea"
)

if st.button("🔍 በሽታውን ይተንብዩ / Predict Disease"):
    if user_input:
        patient_symptoms = [s.strip() for s in user_input.split(',')]
        patient_symptoms = [correct_spelling(symptom) for symptom in patient_symptoms if correct_spelling(symptom)]
        if patient_symptoms:
            predicted_disease = predicted_value(patient_symptoms)
            dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)
            
            st.success(f"**🇪🇹 የተተነበየ በሽታ / Predicted Disease:** {predicted_disease}")
            
            # Create columns for better organization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📝 መግለጫ / Description**")
                st.info(dis_des)
                
                st.markdown("**🛡️ ጥንቃቄዎች / Precautions**")
                st.write(', '.join(str(item) for item in precautions if item))
            
            with col2:
                st.markdown("**💊 መድሃኒቶች / Medications**")
                st.write(', '.join(str(item) for item in medications if item))
                
                st.markdown("**🥗 የሚመከር አመጋገብ / Recommended Diet**")
                st.write(', '.join(str(item) for item in rec_diet if item))
                
                st.markdown("**🏋️ የሚመከር የአካል ብቃት እንቅስቃሴ / Recommended Workout**")
                st.write(', '.join(str(item) for item in workout if item))
        else:
            st.error("🇪🇹 የተሳሳቱ ምልክቶች ተገኝተዋል። እባክዎ ደግመው ይሞክሩ። / Invalid symptoms detected. Please check and try again.")
    else:
        st.warning("🇪🇹 እባክዎ ቢያንስ አንድ ምልክት ያስገቡ። / Please enter at least one symptom.")

st.markdown("---")

# Disease Recommendations Section - Bilingual
st.markdown("""
<div class='bilingual-header'>
    🔍 የበሽታ መግለጫ እና ምክሮች ይፈልጉ <br>
    <small>Search for Disease Description & Recommendations</small>
</div>
""", unsafe_allow_html=True)

disease_query = st.text_input(
    "🇪🇹 የበሽታ ስም ይተይቡ / Type a disease name:", 
    placeholder="መተየብ ይጀምሩ / Start typing..."
)

if disease_query:
    matches = [d for d in disease_names if d.lower().startswith(disease_query.lower())]
    if matches:
        selected_disease = matches[0]
        dis_des, precautions, medications, rec_diet, workout = information(selected_disease)
        
        st.subheader(f"🇪🇹 ለ {selected_disease} የሚመከሩ ነገሮች / Recommendations for {selected_disease}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 መግለጫ / Description**")
            st.info(dis_des)
            
            st.markdown("**🛡️ ጥንቃቄዎች / Precautions**")
            st.write(', '.join(str(item) for item in precautions if item))
        
        with col2:
            st.markdown("**💊 መድሃኒቶች / Medications**")
            st.write(', '.join(str(item) for item in medications if item))
            
            st.markdown("**🥗 የሚመከር አመጋገብ / Recommended Diet**")
            st.write(', '.join(str(item) for item in rec_diet if item))
            
            st.markdown("**🏋️ የሚመከር የአካል ብቃት እንቅስቃሴ / Recommended Workout**")
            st.write(', '.join(str(item) for item in workout if item))
    else:
        st.warning("🇪🇹 ምንም የሚገጥም በሽታ አልተገኘም / No matching disease found")

# Footer with Amharic and English
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px;'>
    <p style='color: white; font-size: 18px;'>
        🇪🇹 ይህ ሥርዓት ለመረጃ ዓላማ ብቻ ነው። እባክዎ ሙያዊ የሕክምና ምክር ይጠይቁ።<br>
        🇬🇧 This system is for informational purposes only. Please consult a healthcare professional for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)