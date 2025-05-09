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
def load_models():
    # Load leaf classifier model
    leaf_classifier = tf.keras.models.load_model('leaf_classifier_model.h5')
    
    # Load disease detection model
    model_path = 'model.tflite'
    fallback_paths = [
        './models/model.tflite',
        './plant-doctor/model.tflite'
    ]

    for path in [model_path] + fallback_paths:
        if os.path.exists(path):
            st.info(f"TFLite model loaded from: {path}")
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return leaf_classifier, interpreter

    error_message = "TFLite model file not found in expected paths."
    st.error(error_message)
    raise FileNotFoundError(error_message)

@st.cache_data
def load_knowledge():
    with open('final_crop_disease_knowledge_base.json') as f:
        return json.load(f)['diseases']

@st.cache_data
def load_class_indices():
    with open('class_indices.json') as f:
        return json.load(f)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

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
        leaf_classifier, interpreter = load_models()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = preprocess_image(img)

        # First stage: Leaf classification
        leaf_pred = leaf_classifier.predict(img_array)
        leaf_class = np.argmax(leaf_pred)
        leaf_confidence = leaf_pred[0][leaf_class]
        
        # Leaf class mapping (update these based on your leaf classifier's class indices)
        LEAF_CLASSES = {
            0: 'non_leaf',
            1: 'other_leaf',
            2: 'tomato_leaf'
        }
        
        current_leaf_class = LEAF_CLASSES[leaf_class]
        
        if current_leaf_class != 'tomato_leaf':
            if current_leaf_class == 'non_leaf':
                st.error("❌ This doesn't appear to be a leaf image. Please upload a clear photo of a tomato leaf.")
            else:
                st.error("❌ This appears to be a non-tomato leaf. Please upload a tomato leaf for disease diagnosis.")
            st.image(img, width=300)
            st.write(f"Classification: {current_leaf_class.replace('_', ' ').title()} ({leaf_confidence*100:.1f}% confidence)")
            return

        # Only proceed with disease detection if it's a tomato leaf
        st.success("✓ Verified: Tomato leaf detected")
        
        # Second stage: Disease detection
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        with st.spinner("🔍 Analyzing for diseases..."):
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

        class_idx = int(np.argmax(output))
        predicted_class = class_indices[str(class_idx)]
        info = knowledge[predicted_class]
        confidence = float(output[class_idx])

        st.image(img, width=300)
        display_results(predicted_class, info, confidence)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
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
