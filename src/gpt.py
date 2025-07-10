import base64
import requests

import os

import tqdm
import csv

from params import parse_args

# OpenAI API Key
api_key = "{Your API}"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def run_gpt(img, result_path, p):
    # Getting the base64 string
    base64_image = []
    for i in tqdm.tqdm(img):
        base64_image.append([i, encode_image(i)])

    idx = 0

    while True:
        try:
            store = []

            for i in tqdm.tqdm(base64_image[idx:]):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                payload = {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "text",
                            "text": p},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{i[1]}"
                            }
                        }
                        ]
                    }
                    ],
                    "max_tokens": 1000
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                store.append([i[0], response.json()['choices'][0]['message']['content']])
                idx+=1
        except:
            with open(result_path+f"gpt_4o_{idx}.csv", "w") as file:
                writer = csv.writer(file) 
                writer.writerows(store)
            continue
        break
    with open(result_path+f"gpt_4o_{idx}.csv", "w") as file:
        writer = csv.writer(file) 
        writer.writerows(store)

def main(args):
    
    result_path = "/data/MWSC/result/"

    prompts = "Within the given class labels, determine the weather in the image and its severity.\
Multiple weather classes may be selected, but only one severity level can be chosen.\
- Weather class labels: [Clear, Foggy, Rainy, Snowy]\
- Severity labels: [Light, Moderate, Heavy]\
The criteria for determining severity are as follows:\
- [Clear class]: Light\
- [Foggy class]\
    200m < visibility < 400m : Light\
    100m < visibility < 200m : Moderate\
    visibility < 100m : Heavy\
- [Rainy and Snowy class]\
    visibility > 1000m : Light\
    500m < visibility < 1000m : Moderate\
    visibility < 500m : Heavy\
The output must strictly follow this format:\
- weather_class : [Weather Classification Result]\
- weather_severity : [Severity Classification Result]"

    dense_root = "/data/MWSC/data/all"

    image = []
    label_name = os.listdir(dense_root)
    for i in label_name:
        img_path = os.path.join(dense_root, i)
        for j in os.listdir(img_path):
            image.append(img_path + '/' + j)
    image.sort()

    run_gpt(image, result_path, prompts)

if __name__ == "__main__":
    args = parse_args()
    main(args)