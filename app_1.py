import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os

# Configure page
st.set_page_config(
    page_title="Plant Disease Doctor",
    page_icon="🌱",
    layout="centered"
)
@st.cache_resource
def load_model():
    model_path = 'tomato_disease_mobilenetv2_finetuned.keras'  # Main model file
    fallback_paths = [
        './models/tomato_disease_mobilenetv2_finetuned.keras',
        './plant-doctor/tomato_disease_mobilenetv2_finetuned.keras'
    ]

    # Try the main path first
    if os.path.exists(model_path):
        st.info(f"Model loaded from: {model_path}")
        return tf.keras.models.load_model(model_path)

    # Try fallback paths
    for path in fallback_paths:
        if os.path.exists(path):
            st.info(f"Model loaded from fallback path: {path}")
            return tf.keras.models.load_model(path)

    # If model not found
    error_message = f"Model file not found at {model_path} or fallback locations."
    st.error(error_message)
    raise FileNotFoundError(error_message)


###########
@st.cache_data
def load_knowledge():
    with open('final_crop_disease_knowledge_base.json') as f:
        return json.load(f)['diseases']  # Access the diseases dictionary

# Main function
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
        # Load resources
        model = load_model()
        knowledge = load_knowledge()
        
        # Preprocess image
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        with st.spinner("🔍 Analyzing..."):
            pred = model.predict(img_array, verbose=0)
            class_idx = np.argmax(pred[0])
            predicted_class = list(knowledge.keys())[class_idx]
            info = knowledge[predicted_class]
            confidence = float(pred[0][class_idx])

        # Display results
        st.image(img, width=300)
        display_results(predicted_class, info, confidence)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

def display_results(predicted_class, info, confidence):
    # Extract plant type (before ___)
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()
    
    if 'healthy' in predicted_class.lower():
        # Healthy plant display
        st.balloons()
        st.success(f"✅ Healthy {plant_type}")
        st.markdown(f"""
        ### Recommendations
        {info['recommendations']}
        
        ### Monitoring Advice
        {''.join([f'- {item}\n' for item in info['monitoring_advice']])}
        """)       
    else:
        # Diseased plant display
        disease_name = predicted_class.split('___')[1].replace('_', ' ').title()
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
