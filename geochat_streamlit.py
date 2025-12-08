import argparse
import os
import random
import re
import math
import numpy as np
from PIL import Image
import torch
import html
import streamlit as st
import sys
import requests
import json
import cv2
from collections import defaultdict

# Core GeoChat imports
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path, process_images_demo

# ==========================================
# 1. Configuration
# ==========================================

class Args:
    def __init__(self):
        self.model_path = "geochat-7B" 
        self.model_base = None
        self.gpu_id = "0"
        self.device = "cuda"
        self.conv_mode = None
        self.max_new_tokens = 300
        self.load_8bit = False
        self.load_4bit = True 
        self.debug = False
        self.image_aspect_ratio = 'pad'

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="geochat-7B")
    parser.add_argument("--load-4bit", action="store_true")
    cli_args, unknown = parser.parse_known_args()
    args = Args()
    args.model_path = cli_args.model_path
    args.load_4bit = cli_args.load_4bit
else:
    args = Args()

# ==========================================
# 2. THE CRITICAL FIXES: SafeChat Class
# ==========================================

class SafeChat(Chat):
    # FIX 1: Force Vision Input to Float32
    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        
        if isinstance(image, str):
            raw_image = Image.open(image).convert('RGB')
            image = process_images_demo([raw_image], self.vis_processor)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = process_images_demo([raw_image], self.vis_processor)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        
        image = image.to(device=self.device, dtype=torch.float32) 
        img_list.append(image)

    # FIX 2: Force Text Generation to Handle Mixed Precision (Float/Half mismatch)
    def model_generate(self, *args, **kwargs):
        with torch.inference_mode():
            # This 'autocast' line is what prevents your current crash
            with torch.cuda.amp.autocast(): 
                output = self.model.generate(
                    kwargs['kwargs']['input_ids'],
                    images=kwargs['kwargs']['images'],
                    do_sample=False,
                    temperature=kwargs['kwargs']['temperature'],
                    max_new_tokens=kwargs['kwargs']['max_new_tokens'],
                    streamer=kwargs['kwargs']['streamer'],
                    use_cache=kwargs['kwargs']['use_cache'],
                    stopping_criteria=kwargs['kwargs']['stopping_criteria']
                )
                if kwargs['kwargs']['streamer'] is None:
                    outputs = self.tokenizer.decode(output[0, kwargs['kwargs']['input_ids'].shape[1]:]).strip()
                    return outputs
                return output

# ==========================================
# 3. Helper Functions (Ollama & Vis)
# ==========================================

def call_ollama_refiner(user_query, vicuna_context, history_list):
    ollama_url = "http://host.docker.internal:11434/api/generate"
    
    # 1. Format History
    history_text = ""
    for msg in history_list[:-1]: 
        role = "USER" if msg["role"] == "user" else "AI"
        clean_content = msg["content"] # content is now just the clean answer
        history_text += f"{role}: {clean_content}\n"

    system_prompt = (
    "You are a senior Geospatial Intelligence Analyst. Your answers must be grounded ONLY in the "
    "visual data provided. If the raw visual context does not explicitly mention something, you must "
    "NOT assume it is absent. Instead, state: 'The detection data does not mention additional objects, "
    "but the image may still contain background elements such as vegetation, soil, water, or terrain.'\n\n"

    "You must ALWAYS describe visible background categories such as vegetation, land, soil, water, "
    "buildings, roads, or terrain IF they appear in the image, even if they were not included in the object list.\n\n"

    "Never give absolute statements like 'there are no other objects' unless the detection data explicitly "
    "lists all objects AND confirms none others exist. When unsure, answer cautiously.\n\n"

    "Your goal is to be accurate, evidence-based, and never overconfident. "
)


    user_prompt = (
    f"--- CONVERSATION HISTORY ---\n{history_text}\n\n"
    f"--- RAW VISUAL DATA (Ground Truth from Vision Model) ---\n{vicuna_context}\n\n"
    f"--- USER QUESTION ---\n{user_query}\n\n"
    f"--- TASK ---\n"
    f"1. Answer the user's question strictly based on the Raw Visual Data. No assumptions, no hallucinations.\n"
    f"2. If the question is straightforward (e.g., a count), give the exact answer concisely.\n"
    f"3. Provide descriptive context ONLY when the image truly contains meaningful visual structure "
    f"(e.g., patterns, density, land-use types, spatial arrangement), and ONLY describe what is explicitly present.\n"
    f"4. Do NOT fabricate details. If something is not visible or not certain, state that clearly.\n"
    f"5. When many similar objects appear (e.g., several planes), summarize their arrangement instead of describing each one.\n"
    f"6. Tone: Professional, factual, precise, and grounded in observable evidence."
)


    payload = {
        "model": "gpt-oss:20b", # Ensure this matches your list
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {"temperature": 0.7}
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error from Ollama: {response.status_code}"
    except Exception as e:
        return f"[System]: Could not connect to Ollama. Error: {e}"

# --- Visualization Logic ---
bounding_box_size = 100
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (210, 210, 0), (255, 0, 255), (0, 255, 255),
    (114, 128, 250), (0, 165, 255), (0, 128, 0), (144, 238, 144), (238, 238, 175),
    (255, 191, 0), (0, 128, 0), (226, 43, 138), (255, 0, 255), (0, 215, 255),
]

def rotate_bbox(top_right, bottom_left, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    center = ((top_right[0] + bottom_left[0]) / 2, (top_right[1] + bottom_left[1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
    rectangle_points = np.array([[bottom_left[0], bottom_left[1]], [top_right[0], bottom_left[1]], 
                                 [top_right[0], top_right[1]], [bottom_left[0], top_right[1]]], dtype=np.float32)
    return cv2.transform(np.array([rectangle_points]), rotation_matrix)[0]

def extract_substrings(string):
    index = string.rfind('}')
    if index != -1: string = string[:index + 1]
    pattern = r'<p>(.*?)\}(?!<)'
    return re.findall(pattern, string)

def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3); intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4); intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    return 0 if union_area == 0 else intersection_area / union_area

def visualize_all_bbox_together(image, generation):
    if image is None: return None, ''
    generation = html.unescape(generation)
    if isinstance(image, Image.Image): image = np.array(image)
    new_image = image.copy()
    image_height, image_width = image.shape[:2]
    
    string_list = extract_substrings(generation)
    mode = 'normal'
    entities = defaultdict(list)
    
    if string_list:
        mode = 'all'
        for string in string_list:
            try: obj, coords = string.split('</p>')
            except: continue
            if "}{" in coords: coords = coords.replace("}{", "}<delim>{")
            for bbox_str in coords.split('<delim>'):
                nums = re.findall(r'-?\d+', bbox_str)
                if len(nums) >= 4:
                    x0, y0, x1, y1 = map(int, nums[:4])
                    angle = int(nums[4]) if len(nums) > 4 else 0
                    l = x0 / 100 * image_width
                    b = y0 / 100 * image_height
                    r = x1 / 100 * image_width
                    t = y1 / 100 * image_height
                    entities[obj].append([l, b, r, t, angle])

    if len(entities) == 0: return None, ''

    text_size, text_line = 0.4, 1
    previous_bboxes = []
    used_colors = colors 
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)

    for entity_idx, entity_name in enumerate(entities):
        bboxes = entities[entity_name]
        for bbox_id, (x1, y1, x2, y2, angle) in enumerate(bboxes):
            orig_x1, orig_y1, orig_x2, orig_y2, angle = int(x1), int(y1), int(x2), int(y2), int(angle)
            color = used_colors[entity_idx % len(used_colors)]
            top_right = (orig_x1, orig_y1)
            bottom_left = (orig_x2, orig_y2)
            rotated_bbox = rotate_bbox(top_right, bottom_left, angle)
            cv2.polylines(new_image, [rotated_bbox.astype(np.int32)], isClosed=True, thickness=2, color=color)

            if mode == 'all':
                # Simplified Label Logic
                x1_lbl, y1_lbl = orig_x1, orig_y1 - 5
                cv2.putText(new_image, f" {entity_name}", (x1_lbl, y1_lbl), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA)

    generation_colored = generation
    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors: yield color
        color_gen = color_iterator(colors)
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            hex_color = f"#{hex(color[0])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[2])[2:].zfill(2)}"
            return f'<span style="color:{hex_color}">{phrase}</span>'
        
        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)

    return Image.fromarray(new_image), generation_colored

# ==========================================
# 4. Model Loading
# ==========================================
@st.cache_resource
def load_geochat_model():
    print('Loading Geo-Intellix Model...')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, device=args.device
    )
    
    # 🧠 Vision tower in Float32 (your existing fix)
    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to(dtype=torch.float16)

    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, 'to'):
            vision_tower.to(dtype=torch.float32)
    elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        model.model.vision_tower.to(dtype=torch.float32)

    # Use SafeChat with Autocast
    chat_model = SafeChat(model, image_processor, tokenizer, device='cuda:0')
    return chat_model


# ==========================================
# 5. Streamlit App Logic
# ==========================================
st.set_page_config(page_title="GeoChat + GPT-OSS", layout="wide")
st.title("🛰️ Geo-Intellix: Multi-Engine Analysis")

with st.sidebar:
    st.header("Refinement Engine")
    use_refiner = st.checkbox("Enable GPT-OSS Refinement", value=False)

try:
    chat = load_geochat_model()
except Exception as e:
    st.error(f"Failed to load Geo-Intellix: {e}")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "chat_state" not in st.session_state: st.session_state.chat_state = None
if "img_list" not in st.session_state: st.session_state.img_list = []

# --- 1. Display History ---
if st.session_state.chat_state:
    history = st.session_state.chat_state.to_streamlit_history()
    for msg in history:
        with st.chat_message(msg["role"]):
            if "image" in msg and msg["image"]:
                st.image(msg["image"], width=300)
            if msg["content"]:
                # Simply display the content. 
                # Since we saved only the smart answer above, that's all that will show here.
                st.markdown(msg["content"], unsafe_allow_html=True)

# Input Handling
uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    if "curr_file" not in st.session_state or st.session_state.curr_file != uploaded_file.name:
        st.session_state.curr_file = uploaded_file.name
        st.session_state.chat_state = conv_templates['llava_v1'].copy()
        st.session_state.img_list = []
        st.session_state.messages = []
        img = Image.open(uploaded_file).convert('RGB')
        chat.upload_img(img, st.session_state.chat_state, st.session_state.img_list)
        st.success("Image Encoded.")

    user_input = st.chat_input("Ask about this area...")
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        augmented_query = f"Describe the scene in this image in detail, covering land use, objects, and density. Then, answer this specific question: {user_input}" if use_refiner else user_input

        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown("Thinking (Geo-Intellix Vision)...")
            
            if len(st.session_state.img_list) > 0 and not isinstance(st.session_state.img_list[0], torch.Tensor):
                chat.encode_img(st.session_state.img_list)
                
            chat.ask(augmented_query, st.session_state.chat_state)
            
            vicuna_stream = chat.stream_answer(
                conv=st.session_state.chat_state,
                img_list=st.session_state.img_list,
                max_new_tokens=500
            )
            
            full_vicuna_response = ""
            for chunk in vicuna_stream:
                full_vicuna_response += chunk
                status.markdown(full_vicuna_response + "▌")
            
            # Check Visual Grounding
            visual_img = None
            if '<p>' in full_vicuna_response:
                try:
                    visual_img, colored_text = visualize_all_bbox_together(img, full_vicuna_response)
                    if visual_img:
                        st.image(visual_img, caption="Detected Regions")
                        full_vicuna_response = colored_text 
                except Exception as e:
                    print(f"Vis Error: {e}")

            if use_refiner:
                status.markdown(f"{full_vicuna_response}\n\n--- \n*Refining with GPT-OSS...*")
                final_answer = call_ollama_refiner(user_input, full_vicuna_response,st.session_state.chat_state.to_streamlit_history())
                combined_display = f"**👁️ Raw Vision Context:**\n{full_vicuna_response}\n\n🧠 GPT-OSS Insight:\n\n{final_answer}"
                status.markdown(combined_display, unsafe_allow_html=True)
                st.session_state.chat_state.messages[-1][1] = combined_display
            else:
                status.markdown(full_vicuna_response, unsafe_allow_html=True)
                st.session_state.chat_state.messages[-1][1] = full_vicuna_response
                
            st.rerun()

elif not uploaded_file:
    st.info("Please upload an image to begin.")