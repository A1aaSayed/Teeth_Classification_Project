import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from utils import predict_image

# ---------------- CONFIG ----------------
st.set_page_config(
  page_title='DentalAI Diagnostics',
  page_icon='🦷',
  layout='wide',
)

# ---------------- CUSTOM CSS ----------------
st.markdown('''
<style>
  .main {
      background-color: #f5f7f9;
      }
  .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
''', unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
  return tf.keras.models.load_model(
    '../results/models/EfficientNet_best_model.keras',
    compile=False
    )

model = load_model()
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# ---------------- SIDEBAR ----------------
with st.sidebar:
  st.image('https://cdn-icons-png.flaticon.com/512/3467/3467830.png', width=100)
  st.title('DentalAI Panel')
  st.info('This AI diagnostic tool helps identify 7 dental conditions with 99.6% accuracy.')
  st.divider()
  st.markdown('### How to use')
  st.write("1. Upload a clear dental image.")
  st.write("2. Wait for Preprocessing.")
  st.write("3. View AI diagnostic report.")

# ---------------- UI ----------------
st.title('🦷 Dental Disease Diagnostic System')
st.markdown("---")

col1, col2 = st.columns([1, 1.2]) # Split screen into 2 columns
# st.write('Upload a dental image and get the predicted tooth class')

with col1:
  st.subheader('📸 Image Upload')
  uploaded_file = st.file_uploader('', type=['jpg', 'png', 'jpeg'])

  if uploaded_file:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    st.image(image, caption='Original Image', use_container_width=False)

    predict_btn = st.button('✨ Generate Diagnostic Report')

with col2:
  if uploaded_file and predict_btn:
    with st.spinner('🔬 Analyzing patterns...'):
      pred_class, probs = predict_image(model, image, class_names)
    
    st.subheader('📊 Diagnostic Results')

    st.markdown(f"""
            <div class="prediction-card">
                <h3>Primary Diagnosis</h3>
                <h1 style='color: #007bff;'>{pred_class}</h1>
                <p>Confidence: {max(probs)*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    fig = px.bar(
            x=probs, 
            y=class_names, 
            orientation='h',
            labels={'x': 'Probability', 'y': 'Condition'},
            title="Probability Distribution",
            color=probs,
            color_continuous_scale='Blues'
        )
    fig.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

  else:
        st.warning("Please upload a dental image to begin analysis.")
        st.image("https://img.freepik.com/free-vector/dentist-examining-patient-teeth-with-tools_1308-91956.jpg?t=st=1716380000")

  # if st.button('Predict'):
  #   with st.spinner('Predicting...'):
  #     pred_class, probs = predict_image(model, image, class_names)

  #   st.success(f'✅ Predicted Class: **{pred_class}**')

  #   df_probs = pd.DataFrame({'Class': class_names, 'Probability': probs})

  #   # Probability Chart
  #   st.subheader('Prediction Probabilities')
  #   # chart_data = {
  #   #   'Class': class_names,
  #   #   'Probability (%)': probs
  #   # }

  #   st.bar_chart(df_probs.set_index('Class'))
