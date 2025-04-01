import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imageupload import upload_to_imgbb
import base64
import io
from gtts import gTTS
import os
from dotenv import load_dotenv

load_dotenv()
imagebb_api_key = os.getenv("IMGBB_API_KEY")
if not imagebb_api_key:
    raise ValueError("IMGBB_API_KEY environment variable is not set.")
# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load captioning model
model = tf.keras.models.load_model(
    'mymodel.h5',
    custom_objects={'LSTM': tf.keras.layers.LSTM}
)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Upload image to imgbb
    uploaded_url = upload_to_imgbb(uploaded_image, imagebb_api_key)
    proxy = f"https://proxy.duckduckgo.com/iu/?u={uploaded_url}"
    google_search_url = f"https://images.google.com/searchbyimage?safe=off&sbisrc=tg&image_url={proxy}"

    st.markdown(
        f'<a href="{google_search_url}" target="_blank"> Reverse Search</a>',
        unsafe_allow_html=True
    )

    st.subheader("Generated Caption")
    with st.spinner("Generating caption..."):
        # Load and process image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features
        image_features = mobilenet_model.predict(image, verbose=0)

        # Generate caption
        max_caption_length = 34
        
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
            )

        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

    # Display caption
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Generate audio from caption
    try:
        with st.spinner("Generating audio..."):
            tts = gTTS(text=generated_caption, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            # Convert to base64 for embedding
            audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Create audio HTML with autoplay
            audio_html = f"""
            <audio controls autoplay style="margin-top: 20px;">
                <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
                Your browser does not support audio playback.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")