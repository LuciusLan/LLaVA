import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from PIL import Image
import numpy as np
import spacy
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from chair_runtime import CHAIR



chair_evaluator = pickle.load(open('./data/chair.pkl', 'rb'))

def get_object_words(self, caption, image_id):
    words, node_words, idxs = self.caption_to_words_with_l(caption)
    gt_objects = self.imid_to_objects[image_id]
    
    gt_captions_file = './data/captions_val2014.json'
    with open(gt_captions_file) as f:
        gt_captions = json.load(f)
    
    current_gt_captions = list(filter(lambda x:x["image_id"]==image_id,gt_captions['annotations']))
    current_gt_captions = [e['caption'] for e in current_gt_captions]

    gt_objects = list(gt_objects)
    generated_objects =  list(node_words),
    hallu_objects = []
    hallu_pos = []
    tp_objects = []
    tp_pos = [ ]

    for word, node_word, idx in zip(words, node_words, idxs):
        if node_word not in gt_objects:
            hallu_objects.append((word, node_word))
            hallu_pos.append(idx)
        else:
            tp_objects.append(node_word)
            tp_pos.append(idx)
    
    # pos: in syntax of list[position, length]
    return gt_objects, generated_objects, hallu_objects, hallu_pos, tp_objects, tp_pos

def caption_to_words_with_l(self, caption) -> tuple[list[str], list[str], list[tuple[int]]]:

    '''
    Input: caption
    Output: MSCOCO words in the caption
    '''

    #standard preprocessing
    sp = nlp(caption)
    words = [t.text for t in sp.doc]
    ori_words = words.copy()
    tagged_sent = [t.pos_ for t in sp.doc]
    lemmas_sent = []
    for i, tag in enumerate(tagged_sent):
        wordnet_pos = tag
        lemmas_sent.append(sp.doc[i].lemma_)
    # words = [singularize(w) for w in words]
    words = lemmas_sent

    #replace double words
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i) 
        double_word = ' '.join(words[i:i+2])
        if double_word in self.double_word_dict: 
            double_words.append(self.double_word_dict[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words

    #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
    if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

    #get synonyms for all words in the caption

    # edited here to include length of phrases
    gen_obj_idxs = [(idxs[idx], len(word.split(' '))) for idx, word in enumerate(words) \
            if word in set(self.mscoco_objects)]
    words = [word for word in words if word in set(self.mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(self.inverse_synonym_dict[word])
    #return all the MSCOCO objects in the caption

    return words, node_words, gen_obj_idxs


setattr(CHAIR, 'caption_to_words_with_l', caption_to_words_with_l)
setattr(CHAIR, 'get_object_words', get_object_words)


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
    
    
    # MODEL:
    #print(model)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    attn_all_gen, attn_hallu, attn_tp  = [], [] ,[]
    for j, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
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
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            
            output_ids = model.generate(
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
                return_dict_in_generate=True,# CQ: add for attention map
            )
            split_sizes, img_emb_len = model.get_img_emb_split_pos(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), image_sizes=[image.size])
        # CQ: add for attention map
        print(output_ids.keys())
        attentions = output_ids.attentions
        # Tuple (one element for each generated token) of Tuples (one element for each layer of the decoder)
        # Tensor Shape (batch_size, num_heads, generated_length, sequence_length)
        #print(f"Type of attentions object: {type(attentions)}")
        #print(f"Length of sequence: {len(attentions)}")
        # Example: Access the attention weights of the first layer
        #for i  in attentions:
        #    print(len(i))
        #    print(i[0].shape)
        ##
        
        outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
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
        gt_objects, generated_objects, hallu_objects, hallu_pos, tp_objects, tp_pos = chair_evaluator.get_object_words(caption=outputs, image_id=image_id)
        sp = nlp(outputs)
        words = [t.text for t in sp.doc]
        pos_map = llm_to_sp_pos_map(output_ids.sequences.cpu()[0], words, tokenizer)
        target_regions_tp = [get_llm_pos_from_obj(e, pos_map) for e in tp_pos]
        target_regions_hallu = [get_llm_pos_from_obj(e, pos_map) for e in hallu_pos]


        attentions = [[attn_layer.detach().cpu().squeeze(0).mean(0).numpy() for attn_layer in attn_token] for attn_token in attentions]
        attn_tp.append(get_saliency(attentions, split_sizes, img_emb_len, target_regions_tp))
        attn_hallu.append(get_saliency(attentions, split_sizes, img_emb_len, target_regions_hallu))
        attn_all_gen.append(get_saliency(attentions, split_sizes, img_emb_len, [[1,-1]]))
        del output_ids
        torch.cuda.empty_cache()
        gc.collect()
    with open('temp_attn_result', 'wb') as f:
        pickle.dump([attn_tp, attn_hallu, attn_all_gen], f)
    f.close()
    ans_file.close()

def get_saliency(attentions, split_sizes, img_emb_len, target_regions):
    sys_message_end = split_sizes[0]
    img_token_end = split_sizes[0]+img_emb_len
    instruction_end = img_token_end+split_sizes[1]

    instruction_to_output = []
    img_emb_to_output = []
    for region in target_regions:
        # Tuple of len region length (Tuple of num_layers)
        # (generated_length, sequence_length)
        if len(region) == 1:
            area_of_interst = attentions[region[0]]
        else:
            area_of_interst = attentions[region[0]:region[1]]
        for i, area in enumerate(area_of_interst):
            area = np.stack(area)
            instruction_to_output.append(area[:, 0, img_token_end-1:instruction_end-1])
            img_emb_to_output.append(area[:, 0, sys_message_end-1:img_token_end-1])

    return instruction_to_output, img_emb_to_output

def get_llm_pos_from_obj(obj_pos, pos_map):
    #obj_word = output_text[obj_pos[0]:obj_pos[0]+obj_pos[1]]
    counter_lm = -1
    temp = []
    flag = False
    for i, pos in enumerate(pos_map):
        if pos == 1:
            counter_lm += 1
        if counter_lm == obj_pos[0] and not flag:
            temp.append(i)
            flag = True
        if counter_lm == obj_pos[0]+obj_pos[1]:
            temp.append(i)
            break
    return temp

def llm_to_sp_pos_map(output_ids, output_text, tokenizer):
    """
        output_pos is following the tokenization of output_text (from spacy
        output_ids is following LLM's tokenization (subword level)
    """
    output_subwords = tokenizer.convert_ids_to_tokens(output_ids)
    try:
        output_subwords.remove(tokenizer.eos_token)
    except:
        pass
    llm_to_spacy_map = []
    temp = ''
    sp_flag = False
    sp_count = 0
    prev = ''
    for i, subword in enumerate(output_subwords):
        if subword == '<0x0A>':
            subword = '\n'
        if subword.startswith('<'):
            print(f'Special character {subword}')
        if i == 0:
            if subword.startswith('▁') and subword != '▁':
                prev = subword[1:]
            elif subword == '<s>':
                prev = ''
            else:
                raise AttributeError("First token not being start of new word")
            continue
        
        if i == len(output_subwords) - 1:
            if subword.startswith('▁') and subword != '▁':
                prev = subword[1:]
            else:
                prev = subword
            if prev == '▁':
                llm_to_spacy_map.append(-1)
                break
            if sp_flag:
                temp = temp + prev
            else:
                temp = prev
            # current -1 phrase match sp word, counter step
            if sp_word == temp:
                sp_count += 1
                llm_to_spacy_map.append(1)
                temp = ''
                sp_flag = False
            else:
                raise AttributeError("Last subword failed to combine into sp word")
                sp_flag = True
                llm_to_spacy_map.append(0)

        sp_word = output_text[sp_count]
        
        # -2 whole word doesn't match sp word, adding -1 word and try
        if sp_flag:
            temp = temp + prev
        else:
            temp = prev
        # current -1 phrase match sp word, counter step
        if sp_word == temp:
            sp_count += 1
            llm_to_spacy_map.append(1)
            temp = ''
            sp_flag = False
        else:
            sp_flag = True
            llm_to_spacy_map.append(0)


        if subword.startswith('▁') and subword != '▁':
            prev = subword[1:]
        else:
            prev = subword

    return llm_to_spacy_map


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
