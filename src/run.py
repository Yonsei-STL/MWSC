import os
from PIL import Image
import tqdm
import csv
from run_params import parse_args
import math
import numpy as np

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu

from torchvision.transforms.functional import InterpolationMode

from transformers import MllamaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM

from qwen_vl_utils import process_vision_info


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

########################################################################################
# help functions
########################################################################################

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
    
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    image = Image.open(image).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80, 'NVLM-D-72B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    device_map['language_model.model.rotary_emb'] = 0

    return device_map

########################################################################################
# run functions
########################################################################################                
class Qwen2_5_VL_72B_Instruct():
    def __init__(self, result_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            torch_dtype="auto",
            # attn_implementation="flash_attention_2",
            device_map="auto"
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        self.result_path = result_path
        
    def run(self, img, p):
        outputs = []
        cnt = 0
        for i in tqdm.tqdm(img[cnt:]):
            image = Image.open(i).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": p},
                    ],
                }
            ]
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=1000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
            outputs.append((i.split('/')[-1], output_text))
                    
            if cnt % 499 == 0:
                with open(self.result_path+f"Qwen2_5_VL_72B_Instruct_backup{cnt}.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(outputs)
            cnt += 1
            image.close()
            
        with open(self.result_path+"Qwen2_5_VL_72B_Instruct_final.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(outputs)

        torch.cuda.empty_cache()

class InternVL2_5_78B_MPO():
    def __init__(self, result_path):
        path = "OpenGVLab/InternVL2_5-78B-MPO"
        device_map = split_model('InternVL2_5-78B')
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
            low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        self.result_path = result_path
        
    def run(self, img, p):
        outputs = []
        cnt = 0
        for i in tqdm.tqdm(img[cnt:]):
            pixel_values = load_image(i, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1000, do_sample=True)

            question = f'<image>\n{p}'
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

            outputs.append((i.split('/')[-1], response))
            if cnt % 499 == 0:
                with open(self.result_path+f"InternVL2_5_78B_MPO_backup{cnt}.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(outputs)
            cnt += 1
            
        with open(self.result_path+"InternVL2_5_78B_MPO_final.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(outputs)

        torch.cuda.empty_cache()
                
class llava_next_72b_hf():
    def __init__(self, result_path):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-next-72b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-next-72b-hf", torch_dtype=torch.float16, device_map="auto") 
        self.result_path = result_path
        
    def run(self, img, p):
        outputs = []
        cnt = 0
        for i in tqdm.tqdm(img[cnt:]):
            conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {"type": "image"},
                        ],
                    },
                ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = self.processor(images=Image.open(i).convert("RGB"), text=prompt, return_tensors="pt").to(self.model.device)

            # autoregressively complete prompt
            output = self.model.generate(**inputs, max_new_tokens=1000)

            outputs.append((i.split('/')[-1], self.processor.decode(output[0], skip_special_tokens=True)))
            if cnt % 499 == 0:
                with open(self.result_path+f"llava_next_72b_hf_backup{cnt}.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(outputs)
            cnt += 1
            
        with open(self.result_path+"llava_next_72b_hf_final.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(outputs)
        
        torch.cuda.empty_cache()

class llama_3_2_90B_Vision_Instruct():
    def __init__(self, result_path):
        model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.result_path = result_path
        
    def run(self, img, p):
        outputs = []
        cnt = 0
        for i in tqdm.tqdm(img[cnt:]):
            image = Image.open(i).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": p}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=1000)

            outputs.append((i.split('/')[-1], self.processor.decode(output[0])))
            if cnt % 499 == 0:
                with open(self.result_path+f"llama_3_2_90B_Vision_Instruct_backup{cnt}.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(outputs)
            cnt += 1
        with open(self.result_path+"llama_3_2_90B_Vision_Instruct_final.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(outputs)
        
        torch.cuda.empty_cache()

class Ovis2_34B():
    def __init__(self, result_path):
        self.model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-34B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.result_path = result_path
        
    def run(self, img, p):
        outputs = []
        cnt = 0
        for i in tqdm.tqdm(img[cnt:]):
            query = f'<image>\n{p}'
            images = [Image.open(i).convert("RGB")]
            max_partition = 9
            
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
            pixel_values = [pixel_values]
            
            # generate output
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1000,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
                outputs.append((i.split('/')[-1], i[0].split('/')[-1], output))
            if cnt % 499 == 0:
                with open(self.result_path+f"Ovis2_34B_backup{cnt}.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerows(outputs)
            cnt += 1
        with open(self.result_path+"Ovis2_34B_final.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(outputs)
        
        torch.cuda.empty_cache()

########################################################################################
# run model
########################################################################################
def load_model(model_num, result_path):
        
    # big size model
    if model_num == 1:
        print("load_Qwen2_5_VL_72B_Instruct")
        model = Qwen2_5_VL_72B_Instruct(result_path)
    elif model_num == 2:
        print("load_InternVL2_5_78B_MPO")
        model = InternVL2_5_78B_MPO(result_path)
    elif model_num == 3:
        print("load_llava_next_72b_hf")
        model = llava_next_72b_hf(result_path)
    elif model_num == 4:
        print("load_llama_3_2_90B_Vision_Instruct")
        model = llama_3_2_90B_Vision_Instruct(result_path)
    elif model_num == 5:
        print("load_Ovis2_34B")
        model = Ovis2_34B(result_path)
    return model

########################################################################################
# main functions
########################################################################################
def main(args):

    torch.cuda.empty_cache()

    model_num = args.model_num
    result_path = "/data/MWSC/result/"

    model = load_model(model_num, result_path)

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
    model.run(image, prompts)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)