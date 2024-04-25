import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['HF_HOME'] = '/home/wuyin/huggingface_cache/'
import sys
sys.path.insert(1, os.getcwd())
import json
import pickle
from typing import Any
import gc

from tqdm import tqdm
#import shortuuid
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle
#import spacy
#spacy.prefer_gpu()
#nlp = spacy.load('en_core_web_trf')


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
#from llava.grad_analysis import start_save, end_save, get_result, add_activation, add_activation_grad



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #CQ: change for attention map, need eager not sdpa
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="eager", output_attentions=True) # CQ: add for attention map
    
    loss_fn = CrossEntropyLoss()
    
    # MODEL:
    #print(model)
    for p in model.parameters():
        p.requires_grad = False
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    saliency_scores = []

    for i_img, line in enumerate(tqdm(questions)):
        current_saliency = []
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        label = line["label"]
        cur_prompt = qs
        qs += " Please answer with \"yes\" or \"no\"."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0].half()

        with torch.inference_mode():
            
            '''output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                output_attentions=True,
                output_scores=False,
                return_dict_in_generate=True,# CQ: add for attention map
            )'''

            logits = model.forward(
                input_ids,
                images=image_tensor.unsqueeze(0).cuda(),
                image_sizes=[image.size],
                output_attentions=True,
                return_dict=True)

            split_sizes, img_emb_len = model.get_img_emb_split_pos(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), image_sizes=[image.size])
        # CQ: add for attention map
        #print(output_ids.keys())
        next_word_logit = logits.logits[0, -1]
        attentions = logits.attentions
        del logits
        # attentions is a tuple where each item represents the attention weights from a layer
        #print(f"Type of attentions object: {type(attentions)}")
        #print(f"Number of sequence: {len(attentions)}")
        # Example: Access the attention weights of the first layer
        #for i  in attentions:
        #    print(len(i))
        #    print(i[0].shape)
        ##

        outputs = tokenizer.decode(next_word_logit.argmax())
        '''ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"image_id": image_file,
                                    "question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()'''
        image_id = image_file.replace("COCO_val2014_", "").replace(".jpg", "")
        image_id = int(image_id)

        label_set = [label, label[0].upper()+label[1:], label]
        label_set = [torch.LongTensor(tokenizer.encode(lab)[1:]).to(next_word_logit.device) for lab in label_set]

        #loss_set = [loss_fn(next_word_logit.unsqueeze(0), lab) for lab in label_set]
        #min_loss = min(loss_set)
        #min_loss.backward()
        #attentions = [attn.grad() for attn in attentions]
        attentions = [attn.squeeze(0).mean(0) for attn in attentions]
        
        for i in range(32):
            current_saliency.append([get_saliency(attentions[i], split_sizes), outputs.lower()==label])
        saliency_scores.append(current_saliency)

            

        del attentions
        torch.cuda.empty_cache()
        gc.collect()
    with open('saliency_scores.bin', 'wb') as f:
        pickle.dump(saliency_scores, f)
        f.close()
    #ans_file.close()

def get_saliency(attention_mat, split_sizes):
    attention_mat = attention_mat.detach().cpu().clone().numpy()
    np.fill_diagonal(attention_mat, 0)
    instruction_to_output = attention_mat[-1,-split_sizes[1]-1:-1]
    img_emb_to_output = attention_mat[-1, split_sizes[0]-1:-split_sizes[1]-1]

    return instruction_to_output, img_emb_to_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
