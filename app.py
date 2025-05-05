import streamlit as st
import numpy as np
import pickle

from imageupload import upload_to_imgbb
import base64
import io
from gtts import gTTS
import os
from dotenv import load_dotenv
from imageupload import upload_to_imgbb
from generate import predict_step

load_dotenv()
imagebb_api_key = os.getenv("IMGBB_API_KEY")
if not imagebb_api_key:
    raise ValueError("IMGBB_API_KEY environment variable is not set.")


# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Upload image to imgbb
    uploaded_url, local_img_path = upload_to_imgbb(uploaded_image, imagebb_api_key)
    proxy = f"https://proxy.duckduckgo.com/iu/?u={uploaded_url}"
    google_search_url = (
        f"https://images.google.com/searchbyimage?safe=off&sbisrc=tg&image_url={proxy}"
    )

    st.markdown(
        f'<a href="{google_search_url}" target="_blank"> Reverse Search</a>',
        unsafe_allow_html=True,
    )

    st.subheader("Generated Caption")
    with st.spinner("Generating caption..."):
        caption_generated = predict_step([local_img_path])[0]
        print(caption_generated)
    # Display caption
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{caption_generated}”</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Generate audio from caption
    try:
        with st.spinner("Generating audio..."):
            tts = gTTS(text=caption_generated, lang="en")
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            # Convert to base64 for embedding
            audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")

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
