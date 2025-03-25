import requests
import base64
import time
import argparse
import os


headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token r8_Tlfr4hDZc7kQNBk1WrYVUzwldl27pW94JMw9P',  
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Origin": '**',
    "Access-Control-Allow-Methods": "OPTIONS,POST,GET,PATCH"
}

def run_process(image_path, output_path):
    """
    Enhance the input image using the Replicate API and save the output image.
    Returns the path to the output image.
    """
    
    owner = "cjwbw"
    name = "night-enhancement"

    
    url = f'https://api.replicate.com/v1/models/{owner}/{name}'
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch model details: {response.status_code} - {response.text}")

    data = response.json()
    version = data.get("latest_version", {}).get("id", "")

    
    with open(image_path, "rb") as file:
        image_data = file.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")
    mimetype = "image/jpg"
    data_uri_image = f"data:{mimetype};base64,{base64_data}"

    
    body = {
        "version": version,
        "input": {
            "image": data_uri_image
        }
    }

    
    url = 'https://api.replicate.com/v1/predictions'  
    response = requests.post(url, json=body, headers=headers)
    if response.status_code != 201:
        raise Exception(f"Failed to start prediction: {response.status_code} - {response.text}")

    
    response_data = response.json()
    prediction_id = response_data.get('id', '')
    get_url = f'https://api.replicate.com/v1/predictions/{prediction_id}'

    
    output = verify_image(get_url)
    if not output:
        raise Exception("Failed to get the enhanced image from the API.")

    
    output_response = requests.get(output)
    if output_response.status_code != 200:
        raise Exception(f"Failed to download the enhanced image: {output_response.status_code}")

    with open(output_path, "wb") as file:
        file.write(output_response.content)

    return output_path


def verify_image(get_url):
    """
    Verify the status of the prediction and return the output image URL.
    """
    while True:
        response = requests.get(get_url, headers=headers)
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get('error', ''):
                raise Exception(f"Prediction error: {res_data.get('error')}")
            else:
                output = res_data.get('output', [])
                if output:
                    return output
                else:
                    time.sleep(1)
        else:
            raise Exception(f"Failed to verify prediction: {response.status_code} - {response.text}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Enhance an image using the Replicate API.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the enhanced image.")
    args = parser.parse_args()

    try:
        
        output_image_path = run_process(args.image_path, args.output_path)
        print(f"Enhanced image saved to: {output_image_path}")
    except Exception as e:
        print(f"Error: {e}")