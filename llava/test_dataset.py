import torch
import transformers
import json 
import copy
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from typing import Dict, Optional, Sequence, List
import conversation as conversation_lib
from mm_utils import tokenizer_image_token


data_path = '/lpai/dataset/mllm-dataset/24-09-28-1/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
list_data_dict = json.load(open(data_path, "r"))
i = 0 
sources = list_data_dict[i]
if isinstance(i, int):
    sources = [sources]
assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME



def preprocess_multimodal(
    sources: Sequence[str],
) -> Dict:

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]))

print(sources)


def tokenizer_prompt_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt = prompt.replace('<image>', '').strip()
    input_ids = tokenizer(prompt).input_ids

    if return_tensors is not None:
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            L = input_ids.size(0)
            if L <= tokenizer.model_max_length:
                pad_length = tokenizer.model_max_length - L
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), 'constant', tokenizer.pad_token_id)

            input_ids = input_ids[:tokenizer.model_max_length]
            return input_ids
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_prompt: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    ### add control prompt
    conv_prompt = conversation_lib.conv_vicuna_prompt.copy()
    roles_prompt = {"human": conv_prompt.roles[0]}

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    conversations_prompt = [] ## add control prompt
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        conv_prompt.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

            ## add control prompt
            if sentence["from"] in roles_prompt:
                role_prompt = roles_prompt[sentence["from"]]
                conv_prompt.append_message(role_prompt, sentence["value"])

        conversations.append(conv.get_prompt())
        
        conversations_prompt.append(conv_prompt.get_prompt()) ## add control prompt

    # Tokenize conversations
    
    ##### add control prompt
    prompt_input_ids = torch.stack([tokenizer_prompt_token(prompt, tokenizer_prompt, return_tensors='pt') for prompt in conversations_prompt], dim=0)


    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt = prompt_input_ids, ### add control prompt
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_prompt: transformers.PreTrainedTokenizer = None,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

vicuna_tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/lpai/volumes/so-volume-ga/models/vicuna-7b-v1.5',
    padding_side="right",
    use_fast=False,
)

clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336',
    padding_side="right",
    use_fast=False,
)


data_dict = preprocess_plain(
            sources,
            vicuna_tokenizer,
            clip_tokenizer)

if 'image' in list_data_dict[i]:
    data_dict['image'] = list_data_dict[i]['image']





print(data_dict)