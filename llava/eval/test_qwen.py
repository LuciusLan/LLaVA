import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,3'
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

#import bitsandbytes as bnb
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
#from llava.grad_analysis import start_save, end_save, get_result, add_activation, add_activation_grad


SAVE_INTERVAL = 5

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

    handles = [layer.self_attn.register_forward_hook(forward_hook) for layer in model.transformer.h.attn]
    return handles

def register_attn_layer_gradient(model, grad_list):
    def hook_layers(module, grad_in, grad_out):
        grad_list.append(grad_out[0].detach().cpu().numpy())
    
    hooks = [layer.self_attn.register_full_backward_hook(hook_layers) for layer in model.model.layers]
    return hooks

def make_context(
    tokenizer,
    query: str,
    history: list[tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #CQ: change for attention map, need eager not sdpa
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    model=model.to(torch.float32)

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


    # query = tokenizer.from_list_format([
    #     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
    #     {'text': '这是什么?'},
    # ])
    # response, history = model.chat(tokenizer, query=query, history=None)
    # print(response)

    loss_fn = CrossEntropyLoss()
    
    # MODEL:
    #print(model)
    #for p in model.parameters():
    #    p.requires_grad = False
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding='utf-8')]
    #questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    ni_ans = [json.loads(q) for q in open(os.path.expanduser('checkpoint/output_scores_asap.jsonl'), "r", encoding='utf-8')]

    saliency_scores = []
    #out_file =  open('max_bw.jsonl', 'w')
    for i_img, (line, nia) in tqdm(enumerate(zip(questions, ni_ans)), total=len(questions)):
        if i_img > 249:
            break
        idx = line['question']['question_id']
        image_id = line['question']['image_id']
        label = line['answer']['multiple_choice_answer']

        ni_pred = nia[0]['token'] if nia[0]['probs'][0] > 0.8 else None

        qs = f"Please answer the question. Give your answer with the answer keyword(s) only, make it concise but accurate.\nQuestion:{line['question']['question']}\nAnswer:"

        
        question_len = len(tokenizer.encode(line['question']['question'])) -1
        image_file = "COCO_val2014_" + "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
        image_file = os.path.join(args.image_folder, image_file)

        query = tokenizer.from_list_format([
            {'image': image_file}, # Either a local path or an url
            {'text': qs},
        ])

        raw_text, input_ids = make_context(
            tokenizer,
            query,
            history=[],
            system="You are a helpful assistant.",
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format,
        )

        #with torch.inference_mode():
        torch.enable_grad()
        model.eval()

        model.zero_grad()
        attn_weight_list = []
        handles = register_attn_layers(model, attn_weight_list)

        #attn_out_grad_list = []
        #hooks = register_attn_layer_gradient(model, attn_out_grad_list)

        input_ids = torch.LongTensor([input_ids], device=model.device)
        logits = model.forward(
            input_ids,
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

        #split_sizes, img_emb_len = model.get_img_emb_split_pos(input_ids, images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(), image_sizes=[image.size])
        bos_pos = torch.where(input_ids == model.config.visual['image_start_id'])
        eos_pos = torch.where(input_ids == model.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        split_sizes = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        img_emb_len = 256

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
        
        #next_word_logit[gt_id].backward()

        #pred_prob = next_word_logit.softmax(0)[next_word_logit.argmax()].detach().cpu().tolist()

        #label_set = [label, label[0].upper()+label[1:]]
        # Only looking at first decoded token, could be a subword (e.g. "bl" in "blonde")
        #label_set = [torch.LongTensor([tokenizer.encode(lab)[1]]).to(next_word_logit.device) for lab in label_set]
        #loss_set = [loss_fn(next_word_logit.unsqueeze(0), lab) for lab in label_set]
        min_loss = min(loss_set)
        #
        #lab = next_word_logit.argsort(descending=True)[0]
        #min_loss = loss_fn(next_word_logit.unsqueeze(0), lab.view(1))
        min_loss.backward()
        #next_word_logit[lab].backward()
        
        logits = None
        [h.remove() for h in handles]
        #[h.remove() for h in hooks]

        saliency = [attn_weight_list[i] * attn_weight_list[i].grad for i in range(len(attn_weight_list))]
        attn_weight_list = None
        saliency = [e.detach().cpu().squeeze(0).abs().mean(dim=0)for e in saliency]
        #saliency = [e.detach().squeeze(0).clamp(min=0).mean(dim=0) for e in saliency] # Mean over 32 heads
        
        #temp = torch.stack(saliency, 0).to(torch.bfloat16)
        
        # query = input_ids.where(input_ids!=-200, 2)
        # query = tokenizer.convert_ids_to_tokens(query[0])
        # temp = [temp, query, question_len, img_emb_len]
        # torch.save(temp, f'mat_{idx}.pt')

        """current_saliency = {}
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
                            }"""
        img_self, img_rest = get_img_sal(saliency, img_emb_len)
        current_saliency = {'qid': idx, 'img_self': img_self, 'img_rest': img_rest}

        '''
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
        '''

        saliency_scores.append(current_saliency)
        #out_file.write(json.dumps(current_saliency))
        #out_file.write('\n')
        saliency = None
        torch.cuda.empty_cache()
        gc.collect()

        if i_img % SAVE_INTERVAL == 0:
            torch.save(saliency_scores, f'./checkpoint/raw_sal/{i_img}.pt', pickle_protocol=pickle.HIGHEST_PROTOCOL)
            saliency_scores = []
    #torch.save(saliency_scores, 'max_bw.pt')
    #out_file.close()

def get_saliency(attention_mat, split_sizes, img_emb_len, question_len):
    attention_mat = attention_mat.fill_diagonal_(0)
    instruction_to_output = attention_mat[-1,-8-question_len:-8]

    # 5 for [INST] at begining
    img_emb_to_output = attention_mat[-1, 5-1:5+img_emb_len-1]

    wholeseq_to_output = attention_mat[-1, :]

    img_to_instruction = attention_mat[-8-question_len:-8, 5-1:5+img_emb_len-1]

    return instruction_to_output, img_emb_to_output, wholeseq_to_output, img_to_instruction

def get_img_sal(attention_mat, img_emb_len):
    attention_mat = [e.detach().fill_diagonal_(0) for e in attention_mat]
    attention_mat = torch.stack(attention_mat, dim=0)

    
    return attention_mat[:, 5:5+img_emb_len, 5:5+img_emb_len].numpy(), attention_mat[:, 5+img_emb_len:, 5:5+img_emb_len].numpy()

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
