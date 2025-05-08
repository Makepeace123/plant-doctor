import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os
import requests
import zipfile
import io

# Configure page
st.set_page_config(
    page_title="Plant Disease Doctor",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_keras_model():
    try:
        # Download and extract the zip file from GitHub
        zip_url = "https://github.com/yourusername/repo/releases/download/v1.0/tomato_doctor_mobilenetv2.zip"
        response = requests.get(zip_url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Extract the model file (assuming it's named 'model.keras' inside the zip)
        model_filename = 'model.keras'
        zip_file.extract(model_filename)
        
        # Load the Keras model
        model = tf.keras.models.load_model(tomato_doctor_mobilenetv2.keras)
        st.success("Keras model loaded successfully from zip archive!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_knowledge():
    try:
        with open('final_crop_disease_knowledge_base.json') as f:
            return json.load(f)['diseases']
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        st.stop()

@st.cache_data
def load_class_indices():
    try:
        with open('class_indices.json') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class indices: {str(e)}")
        st.stop()

def main():
    st.title("🍅🌿 Tomato Disease Diagnosis and Doctor 🔬🩺")
    st.markdown("Upload a clear photo of a plant leaf for instant analysis")

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=[".jpg", ".png", ".jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        process_image(uploaded_file)

def process_image(uploaded_file):
    try:
        model = load_keras_model()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        # Preprocess image
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)

        # Inference
        with st.spinner("🔍 Analyzing..."):
            output = model.predict(input_tensor)[0]

        class_idx = int(np.argmax(output))
        predicted_class = class_indices[str(class_idx)]
        info = knowledge[predicted_class]
        confidence = float(output[class_idx])

        st.image(img, width=300)
        display_results(predicted_class, info, confidence)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.stop()

def display_results(predicted_class, info, confidence):
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()

    if 'healthy' in predicted_class.lower():
        st.balloons()
        st.success(f"✅ Healthy {plant_type}")
        st.markdown(f"""
        ### Recommendations
        {info['recommendations']}
        
        ### Monitoring Advice
        {''.join([f'- {item}\n' for item in info['monitoring_advice']])}
        """)       
    else:
        disease_name = predicted_class.split('___')[1].replace('_', ' ').title() if '___' in predicted_class else predicted_class.replace('_', ' ').title()
        st.warning(f"⚠️ Detected: {disease_name} ({confidence*100:.1f}% confidence)")
        
        tab1, tab2, tab3 = st.tabs(["Symptoms", "Treatment", "Prevention"])
        
        with tab1:
            st.markdown(f"""
            **Plant Type:** {plant_type}
            
            **Symptoms:**  
            {info['symptoms']}
            
            **Causes:**  
            {info['causes']}
            
            **Effects:**  
            {info['effects']}
            """)
        
        with tab2:
            if info['treatments']['chemical']:
                chem = info['treatments']['chemical']
                st.markdown(f"""
                ### Chemical Treatment
                **{chem['product']}**  
                - Dosage: {chem['dosage']}  
                - Instructions: {chem.get('note', 'N/A')}
                """)
            else:
                st.info("No chemical treatment recommended")
            
        with tab3:
            st.markdown("### Cultural Practices")
            for method in info['treatments']['cultural']:
                st.markdown(f"- {method}")

if __name__ == "__main__":
    main()
