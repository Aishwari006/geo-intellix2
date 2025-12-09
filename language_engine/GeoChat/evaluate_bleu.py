import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# GeoChat Imports
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path, process_images_demo, tokenizer_image_token, KeywordsStoppingCriteria
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="geochat-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()

def load_model(args):
    print(f"Loading model from {args.model_path}...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        False, args.load_4bit, device="cuda"
    )
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, 'to'):
            vision_tower.to(dtype=torch.float32)
    elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        model.model.vision_tower.to(dtype=torch.float32)
    return tokenizer, model, image_processor

def calculate_metrics(ground_truth, generated):
    ref_tokens = nltk.word_tokenize(ground_truth.lower())
    cand_tokens = nltk.word_tokenize(generated.lower())
    chencherry = SmoothingFunction()
    w1 = (1.0, 0, 0, 0)
    w2 = (0.5, 0.5, 0, 0)
    w3 = (0.33, 0.33, 0.33, 0)
    w4 = (0.25, 0.25, 0.25, 0.25)
    b1 = sentence_bleu([ref_tokens], cand_tokens, weights=w1, smoothing_function=chencherry.method1)
    b2 = sentence_bleu([ref_tokens], cand_tokens, weights=w2, smoothing_function=chencherry.method1)
    b3 = sentence_bleu([ref_tokens], cand_tokens, weights=w3, smoothing_function=chencherry.method1)
    b4 = sentence_bleu([ref_tokens], cand_tokens, weights=w4, smoothing_function=chencherry.method1)
    return b1, b2, b3, b4

def main():
    args = parse_args()
    tokenizer, model, image_processor = load_model(args)
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    scores = {'b1': [], 'b2': [], 'b3': [], 'b4': []}
    print("Starting Inference...")
    for item in tqdm(data):
        image_file = item['image']
        qs = item['conversations'][0]['value']
        ground_truth = item['conversations'][1]['value']
        if DEFAULT_IMAGE_TOKEN not in qs:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        img_path = os.path.join(args.image_folder, image_file)
        try:
            raw_image = Image.open(img_path).convert('RGB')
            image_tensor = process_images_demo([raw_image], image_processor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                output_ids = model.generate(
                    input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                    max_new_tokens=args.max_new_tokens, use_cache=True, stopping_criteria=[stopping_criteria]
                )
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        b1, b2, b3, b4 = calculate_metrics(ground_truth, outputs)
        scores['b1'].append(b1); scores['b2'].append(b2); scores['b3'].append(b3); scores['b4'].append(b4)
    print("FINAL RESULTS")
    print(f"BLEU-1: {sum(scores['b1'])/len(scores['b1']):.4f}")
    print(f"BLEU-2: {sum(scores['b2'])/len(scores['b2']):.4f}")
    print(f"BLEU-3: {sum(scores['b3'])/len(scores['b3']):.4f}")
    print(f"BLEU-4: {sum(scores['b4'])/len(scores['b4']):.4f}")

if __name__ == "__main__":
    main()