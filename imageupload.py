import requests
import base64
import io

def upload_to_imgbb(image, api_key):
    url = "https://api.imgbb.com/1/upload"
    
    # Check if `image` is a file path (str) or a file-like object (Streamlit UploadedFile)
    if isinstance(image, str):  # If it's a file path
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    else:  # If it's a file-like object
        encoded_image = base64.b64encode(image.getvalue()).decode("utf-8")

    payload = {
        "key": api_key,
        "image": encoded_image,
    }

    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print(response.json())
        return response.json()["data"]["url"]
    else:
        print("Error:", response.json())
        return None
