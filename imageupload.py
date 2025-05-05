import requests
import base64
import io
import os
import time
import random
import string


def save_image_locally(image, save_folder="uploads"):
    os.makedirs(save_folder, exist_ok=True)

    # Generate unique filename
    timestamp = int(time.time())
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    filename = f"{timestamp}_{random_suffix}.jpg"
    save_path = os.path.join(save_folder, filename)

    # Save image from file path or file-like object
    if isinstance(image, str):  # file path
        with open(image, "rb") as src_file, open(save_path, "wb") as dst_file:
            dst_file.write(src_file.read())
    else:  # file-like object
        with open(save_path, "wb") as dst_file:
            dst_file.write(image.getvalue())

    return save_path


def upload_to_imgbb(image, api_key):
    # Save locally
    saved_path = save_image_locally(image)

    # Upload to imgbb
    url = "https://api.imgbb.com/1/upload"
    with open(saved_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "key": api_key,
        "image": encoded_image,
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        upload_url = response.json()["data"]["url"]
        return upload_url, saved_path
    else:
        print("Error:", response.json())
        return None, saved_path
