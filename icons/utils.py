"""This file is for selection utils"""
import os
import json
import copy
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
from PIL import Image

from llava import conversation as conversation_lib
from llava.train.train import *
local_rank = None


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # XINDI: added this for val data
        if isinstance(self.list_data_dict, dict):
            self.list_data_dict = list(self.list_data_dict.values())
        
        sources = self.list_data_dict[i]
            
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except FileNotFoundError as e:
                print(f"Error: {e}. File not found: {image_file}")
                return None  # or handle the error as needed
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and the chat format. """
    
    # First check if raw_datasets is None
    if raw_datasets is None:
        raise ValueError("raw_datasets cannot be None")
    
    # Check if it's a Subset
    if isinstance(raw_datasets, torch.utils.data.Subset):
        if len(raw_datasets) == 0:
            raise ValueError("Dataset is empty")
        if raw_datasets[0] is not None and "input_ids" in raw_datasets[0]:
            return raw_datasets
    # Check if it's a regular dataset
    elif len(raw_datasets) > 0 and raw_datasets[0] is not None and "input_ids" in raw_datasets[0]:
        return raw_datasets
    
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    
    # To speed up this part, we use multiprocessing.
    if isinstance(raw_datasets, torch.utils.data.Subset):
        # Apply the encode function to each element of the subset
        lm_datasets = torch.utils.data.Dataset([encode_function(raw_datasets[i]) for i in range(len(raw_datasets))])
    else:
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=processing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
    
    return lm_datasets

def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    
    if isinstance(raw_datasets, torch.utils.data.Subset):
        # Check if the first element of the subset has "input_ids" and "labels"
        if "input_ids" in raw_datasets[0] and "labels" in raw_datasets[0]:
            # Dataset is already encoded, no need for additional encoding
            return lambda x: x
    elif "input_ids" in raw_datasets.column_names and "labels" in raw_datasets.column_names:
        # Dataset is already encoded, no need for additional encoding
        return lambda x: x
    
    elif "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    
    else:
        raise ValueError(
            "You need to have either 'input_ids'&'labels', 'prompt'&'completion', or 'messages' in your column names.")
    
    return encode_function