import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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


SAVE_INTERVAL = 50

SEED = 123

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def register_attn_layers(model, attn_weight_by_layer):
    def forward_hook(module, inputs, output):
        output[1].retain_grad()
        attn_weight_by_layer.append(output[1])

    handles = [layer.self_attn.register_forward_hook(forward_hook) for layer in model.model.layers]
    return handles

def register_attn_layer_gradient(model, grad_list):
    def hook_layers(module, grad_in, grad_out):
        grad_list.append(grad_out[0].detach().cpu().numpy())
    
    hooks = [layer.self_attn.register_full_backward_hook(hook_layers) for layer in model.model.layers]
    return hooks


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #CQ: change for attention map, need eager not sdpa
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="eager", output_attentions=True, device='cuda', torch_dtype=torch.bfloat16) # CQ: add for attention map
    #model = model.to(torch.float32)
    loss_fn = CrossEntropyLoss()
    
    # MODEL:
    print(model)
    #for p in model.parameters():
    #    p.requires_grad = False
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    #questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    ni_ans = [json.loads(q) for q in open(os.path.expanduser('checkpoint/output_scores_asap.jsonl'), "r")]

    saliency_scores = []
    out_file =  open('saliency_score.jsonl', 'w')
    for i_img, (line, nia) in tqdm(enumerate(zip(questions, ni_ans)), total=len(questions)):
        if i_img > 249:
            break
        idx = line['question']['question_id']
        image_id = line['question']['image_id']
        label = line['answer']['multiple_choice_answer']

        ni_pred = nia[0]['token'] if nia[0]['probs'][0] > 0.8 else None

        qs = f"Please answer the question. Give your answer with the answer keyword(s) only, make it concise but accurate.\nQuestion:{line['question']['question']}\nAnswer:"
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        question_len = len(tokenizer.encode(line['question']['question'])) -1
        question_pos = len(input_ids[0]) - 8 - question_len # -8 for "\nAnswer:[\INST]" have length 8

        image_file = "COCO_val2014_" + "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0].half()

        #with torch.inference_mode():
        torch.enable_grad()
        model.eval()

        model.zero_grad()
        attn_weight_list = []
        handles = register_attn_layers(model, attn_weight_list)

        #attn_out_grad_list = []
        #hooks = register_attn_layer_gradient(model, attn_out_grad_list)

        logits = model.forward(
            input_ids,
            images=image_tensor.unsqueeze(0).cuda(),
            image_sizes=[image.size],
            output_attentions=True,
            return_dict=True)

        # output_ids = model.generate(
        #         input_ids,
        #         attention_mask=None,
        #         images=image_tensor.unsqueeze(0),
        #         image_sizes=[image.size],
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         # no_repeat_ngram_size=3,
        #         max_new_tokens=1024,
        #         output_scores=True,
        #         use_cache=True,
        #         return_dict_in_generate=True,# CQ: add for attention map
        #     )

        split_sizes, img_emb_len = model.get_img_emb_split_pos(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), image_sizes=[image.size])
        #split_sizes, img_emb_len = model.get_img_emb_split_pos(input_ids, images=image_tensor.unsqueeze(0), image_sizes=[image.size])

        next_word_logit = logits.logits[0, -1]
        outputs = tokenizer.decode(next_word_logit.argmax())
               
        pred_tok_id = next_word_logit.argmax().detach().cpu()
        pred_tok = tokenizer.decode(pred_tok_id)

        label_set = [label, label[0].upper()+label[1:]]
        # Only looking at first decoded token, could be a subword (e.g. "bl" in "blonde")
        label_set = [torch.LongTensor([tokenizer.encode(lab)[1]]).to(next_word_logit.device) for lab in label_set]
        loss_set = [loss_fn(next_word_logit.unsqueeze(0), lab) for lab in label_set]
        gt_id = label_set[torch.tensor(loss_set).argmin()]
        
        """
        Grad for the GT token
        """
        
        next_word_logit[gt_id].backward()

        pred_prob = next_word_logit.softmax(0)[next_word_logit.argmax()].detach().cpu().tolist()

        #label_set = [label, label[0].upper()+label[1:]]
        # Only looking at first decoded token, could be a subword (e.g. "bl" in "blonde")
        #label_set = [torch.LongTensor([tokenizer.encode(lab)[1]]).to(next_word_logit.device) for lab in label_set]
        #loss_set = [loss_fn(next_word_logit.unsqueeze(0), lab) for lab in label_set]
        #min_loss = min(loss_set)
        #min_loss.backward()
        
        logits = None
        [h.remove() for h in handles]
        #[h.remove() for h in hooks]

        saliency = [attn_weight_list[i] * attn_weight_list[i].grad for i in range(len(attn_weight_list))]
        saliency = [e.detach().squeeze(0).abs().mean(dim=0) for e in saliency]
        
        current_saliency = {}
        q2o = []
        img2o = []
        whole2o = []
        i2q = []
        for i in range(32):
            temp = get_saliency(saliency[i], split_sizes, img_emb_len, question_len)
            q2o.append(temp[0].sum().tolist())
            img2o.append(temp[1].sum().tolist())
            whole2o.append(temp[2].sum().tolist())
            i2q.append(temp[3].sum().tolist())

            
        current_saliency = {'qid': idx, 'gt': label, 'pred': outputs, 'pred_prob': pred_prob, 'prior':ni_pred, 'qlen':question_len, 'ilen':img_emb_len,
                            'gt_sal':{'q2o': q2o, 'i2o': img2o, 'w2o': whole2o, 'i2q': i2q},
                            'prior_sal': {},
                            'false_pred_sal': {},
                            }
        
        if pred_tok.lower() not in label.lower():
            """
            Grad for the false pred token
            """
            saliency = None

            model.zero_grad()
            attn_weight_list = []
            handles = register_attn_layers(model, attn_weight_list)
            
            logits = model.forward(
                input_ids,
                images=image_tensor.unsqueeze(0).cuda(),
                image_sizes=[image.size],
                output_attentions=True,
                return_dict=True)

            next_word_logit = logits.logits[0, -1]
            logits = None
            [h.remove() for h in handles]

            next_word_logit[pred_tok_id].backward()

            saliency = [attn_weight_list[i] * attn_weight_list[i].grad for i in range(len(attn_weight_list))]
            saliency = [e.detach().squeeze(0).abs().mean(dim=0) for e in saliency]
            
            q2o = []
            img2o = []
            whole2o = []
            i2q = []
            for i in range(32):
                temp = get_saliency(saliency[i], split_sizes, img_emb_len, question_len)
                q2o.append(temp[0].sum().tolist())
                img2o.append(temp[1].sum().tolist())
                whole2o.append(temp[2].sum().tolist())
                i2q.append(temp[3].sum().tolist())

            current_saliency.update({'false_pred_sal': {'q2o': q2o, 'i2o': img2o, 'w2o': whole2o, 'i2q': i2q}})
        if ni_pred is not None:
            if pred_tok.lower() != ni_pred.lower():
                """
                Pred correct
                Prior exists
                Grad for the prior token
                """
                saliency = None

                model.zero_grad()
                attn_weight_list = []
                handles = register_attn_layers(model, attn_weight_list)
            
                logits = model.forward(
                    input_ids,
                    images=image_tensor.unsqueeze(0).cuda(),
                    image_sizes=[image.size],
                    output_attentions=True,
                    return_dict=True)

                
                next_word_logit = logits.logits[0, -1]
                logits = None
                [h.remove() for h in handles]

                prior_id = tokenizer.encode(ni_pred)[1]
                next_word_logit[prior_id].backward()

                saliency = [attn_weight_list[i] * attn_weight_list[i].grad for i in range(len(attn_weight_list))]
                saliency = [e.detach().squeeze(0).abs().mean(dim=0) for e in saliency]

                q2o = []
                img2o = []
                whole2o = []
                i2q = []
                for i in range(32):
                    temp = get_saliency(saliency[i], split_sizes, img_emb_len, question_len)
                    q2o.append(temp[0].sum().tolist())
                    img2o.append(temp[1].sum().tolist())
                    whole2o.append(temp[2].sum().tolist())
                    i2q.append(temp[3].sum().tolist())
                    
                current_saliency.update({'prior_sal':{'q2o': q2o, 'i2o': img2o, 'w2o': whole2o, 'i2q': i2q}})

        #saliency_scores.append(current_saliency)
        out_file.write(json.dumps(current_saliency))
        out_file.write('\n')
        saliency = None
        torch.cuda.empty_cache()
        gc.collect()

        # if i % SAVE_INTERVAL == 0:
        #     with open('saliency_scores.bin', 'wb') as f:
        #         pickle.dump(saliency_scores, f)
        #         f.close()
    out_file.close()

def get_saliency(attention_mat, split_sizes, img_emb_len, question_len):
    attention_mat.fill_diagonal_(0)
    instruction_to_output = attention_mat[-1,-8-question_len:-8]

    # 5 for [INST] at begining
    img_emb_to_output = attention_mat[-1, 5-1:5+img_emb_len-1]

    wholeseq_to_output = attention_mat[-1, :]

    img_to_instruction = attention_mat[-8-question_len:-8, 5-1:5+img_emb_len-1]

    return instruction_to_output, img_emb_to_output, wholeseq_to_output, img_to_instruction

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
