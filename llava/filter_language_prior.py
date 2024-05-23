import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['HF_HOME'] = '/home/wuyin/hf_cache/'
import sys
sys.path.insert(1, os.getcwd())
import json
import pickle
from typing import Any
import gc

from tqdm import tqdm
#import shortuuid
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path




def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #CQ: change for attention map, need eager not sdpa
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name) # CQ: add for attention map
        
    # MODEL:
    #print(model)
    with open('./vqa_color.jsonl') as f:
        questions = f.readlines()
        questions = [json.loads(e) for e in questions]
    f.close()

    #question_texts = [f"[INST]Please answer the question. Give your answer with the answer keyword(s) only, make it as short as possible.\n {e['question']['question']}\nAnswer:[/INST]" for e in questions]
    question_texts = [f"[INST]{e['question']['question']}[/INST]" for e in questions]
    question_ids = tokenizer(question_texts, padding='longest', return_tensors='pt')
    index = torch.LongTensor(np.arange(len(questions))).unsqueeze(1)


    ds = TensorDataset(question_ids.input_ids, question_ids.attention_mask, index)
    dl = DataLoader(ds, batch_size=8)
    output_probs = []
    for batch in tqdm(dl, total=(len(dl))):
        input_ids, attention_mask, index = batch
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=None,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                output_scores=True,
                use_cache=True,
                return_dict_in_generate=True,# CQ: add for attention map
            )
        
        bs = input_ids.size(0)
        scores = output_ids.scores
        scores = torch.stack(scores, dim=1)
        prob = F.softmax(scores, dim=-1)
        top10 = torch.argsort(scores, -1, descending=True)[:,:,:10]

        for i in range(bs):

            pp = prob[i]
            tt = top10[i]
            top10_prob = [p[t] for p, t in zip(pp, tt)]

            output_probs.append([{'token': tokenizer.decode(tok), 'probs':scores.cpu().tolist(), 'top10_tokens': top10_id.cpu().tolist()} for tok, scores, top10_id in zip(output_ids.sequences[i], top10_prob, tt)])

    with open('output_scores.jsonl', 'w') as f:
        for row in output_probs:
            f.write(json.dumps(row, f))
            f.write('\n')
    f.close()



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
