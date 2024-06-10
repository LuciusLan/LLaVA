import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['HF_HOME'] = '/home/wuyin/hf_cache/'
import sys
sys.path.insert(1, os.getcwd())
import json
import pickle
from typing import Any
import gc
import random

from tqdm import tqdm
#import shortuuid
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

SEED = 123

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True



class CustomDataset(Dataset):
    def __init__(self, questions) -> None:
        super().__init__()
        self.questions = questions
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index) -> Any:
        return self.questions[index]['question']['question'], self.questions[index]['question']['image_id']

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #CQ: change for attention map, need eager not sdpa
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name) # CQ: add for attention map
    #model = model.to(torch.float32)
    def custom_collate(batch):
        question_texts = [e[0] for e in batch]
        image_ids = [e[1] for e in batch]
        image_ids = [f"COCO_val2014_{'0' * (12 - len(str(e))) + str(e)}.jpg" for e in image_ids]
        #images = [Image.open(os.path.join(args.image_folder, image_file)).convert('RGB') for image_file in image_ids]
        images = [Image.new('RGB', (336,336), color='white')] * len(image_ids)
        image_sizes = [i.size for i in images]
        images = process_images(images, image_processor, model.config)

        prompts = []
        for qs in question_texts:
            qs = "Please answer the question. Give your answer with the answer keyword(s) only, make it concise but accurate.\nQuestion:" + qs +"\nAnswer:"

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(e[::-1]) for e in input_ids], batch_first=True, padding_value=tokenizer.eos_token_id).flip(dims=[1])
        return input_ids, images, image_sizes
        

    # MODEL:
    #print(model)
    with open('./data/vqa_color.jsonl') as f:
        questions = f.readlines()
        questions = [json.loads(e) for e in questions]
    f.close()

    output_probs = []

    ds = CustomDataset(questions)
    dl = DataLoader(ds, batch_size=8, collate_fn=custom_collate)

    for batch in tqdm(dl, total=len(dl)):
        #qs = question['question']['question']
        #qs = "Please answer the question. Give your answer with the answer keyword(s) only, make it as short as possible.\n" + qs
        #img_id = '0' * (12 - len(str(question['question']['image_id']))) + str(question['question']['image_id'])
        #image_file = f"COCO_val2014_{img_id}"

        #conv = conv_templates[args.conv_mode].copy()
        #conv.append_message(conv.roles[0], qs)
        #conv.append_message(conv.roles[1], None)
        #prompt = conv.get_prompt()

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        #image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        #image_tensor = process_images([image], image_processor, model.config)[0]

        input_ids, images, image_sizes = batch
        input_ids = input_ids.cuda()


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.cuda(),
                images=images.half().cuda(),
                image_sizes=image_sizes,
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

    with open('output_scores_white_img.jsonl', 'w') as f:
        for row in output_probs:
            f.write(json.dumps(row))
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
