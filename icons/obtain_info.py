"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""
import os
import glob
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, HfArgumentParser

from icons.utils import *
from icons.collect_grads import GradientCollector
from llava.model.builder import load_pretrained_model


def save_detailed_model_parameters(model, output_file="detailed_model_parameters.txt"):
    """Save detailed model parameters including actual tensor values to a file."""
    with open(output_file, 'w') as f:
        f.write(f"Detailed Model Parameters Dump - {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        for name, param in model.named_parameters():
            f.write(f"{name}: Parameter containing:\n")
            tensor_str = str(param.data)
            f.write(tensor_str)
            f.write(f"\nShape: {tuple(param.shape)}")
            f.write(f"\nRequires grad: {param.requires_grad}")
            f.write(f"\nDevice: {param.device}")
            f.write(f"\nDtype: {param.dtype}\n")
            f.write("\n" + "-" * 80 + "\n")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class SelectionArguments:
    task: Optional[str] = field(default=None, metadata={"help": "Specify the task from mmvet. One of variables of task and train_file must be specified"})
    train_file: Optional[str] = field(default=None, metadata={"help": "The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified"})
    val_file: Optional[str] = field(default=None, metadata={"help": "The path to the validation data file we'd like to obtain the gradients/representations for"})
    info_type: str = field(default=None, metadata={"help": "The type of information", "choices": ["grads", "reps", "loss", "delta"]})
    model_path: Optional[str] = field(default=None, metadata={"help": "The path to the model"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "The maximum number of samples"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "The torch data type", "choices": ["float32", "bfloat16"]})
    output_path: Optional[str] = field(default=None, metadata={"help": "The path to the output"})
    gradient_projection_dimension: List[int] = field(default_factory=lambda: [5120], metadata={"help": "The dimension of the projection, can be a list"})
    gradient_type: str = field(default="adam", metadata={"help": "The type of gradient", "choices": ["adam", "sign", "sgd"]})
    chat_format: str = field(default="tulu", metadata={"help": "The chat format"})
    use_chat_format: bool = field(default=True, metadata={"help": "Whether to use chat format"})
    max_length: int = field(default=2048, metadata={"help": "The maximum length"})
    zh: bool = field(default=False, metadata={"help": "Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)"})
    lora_r: int = field(default=8, metadata={"help": "The value of lora_r hyperparameter"})
    lora_alpha: float = field(default=32, metadata={"help": "The value of lora_alpha hyperparameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "The value of lora_dropout hyperparameter"})
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"], metadata={"help": "The list of lora_target_modules"})

# Compare specific model parameters
def compare_model_parameters(old_model, new_model):
    for (name1, param1), (name2, param2) in zip(old_model.named_parameters(), new_model.named_parameters()):
        if name1 != name2:
            print(f"Parameter names differ: {name1} vs {name2}")
            return False
        if not torch.equal(param1, param2):
            print(f"Parameters differ for {name1}")
            return False
    print("All specified parameters are identical.")
    return True


def load_raw_dataset(data_args, tokenizer, train_files, sample_percentage=None, seed=None):
    print(f"Train files received: {train_files}")
    
    if train_files is None:
        raise ValueError("train_files cannot be None. Please provide a valid data path.")
    
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_files,
        data_args=data_args
    )
    
    return dataset

def get_dataset(data_args, files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    raw_datasets = load_raw_dataset(data_args, tokenizer, files, sample_percentage=sample_percentage, seed=seed)
    
    class TruncatedDataset:
        def __init__(self, dataset, max_length):
            self.dataset = dataset
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item.get("image", None)
            
            if len(item["input_ids"]) > self.max_length:
                item["input_ids"] = item["input_ids"][:self.max_length]
                if "labels" in item:
                    item["labels"] = item["labels"][:self.max_length]
            
            if not isinstance(item["input_ids"], torch.Tensor):
                item["input_ids"] = torch.tensor(item["input_ids"])
            if "labels" in item and not isinstance(item["labels"], torch.Tensor):
                item["labels"] = torch.tensor(item["labels"])
            
            if image is not None:
                item["image"] = image
            
            return item
    
    return TruncatedDataset(raw_datasets, max_seq_length)

def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    print(f"There are {len(dataset)} examples in the dataset")
    return dataloader

def initialize_model(args):
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    print("Loading model from:", args.model_path)

    if 'lora' in args.model_path:
        if '13b' in args.model_path:
            model_base = "./checkpoints/vicuna-13b-v1.5"
            model_name = "llava-v1.5-13b-lora"
        else:  
            model_base = "./checkpoints/vicuna-7b-v1.5"
            model_name = "llava-v1.5-7b-lora"
    else:
        model_base = None
        model_name = "llava-v1.5-13b" if '13b' in args.model_path else "llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=model_base,  
        model_name=model_name,
        load_8bit=False,
        load_4bit=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if 'lora' in args.model_path:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.to(device)
        print(f"LoRA model loaded successfully and moved to {device}")
        
        save_detailed_model_parameters(model, "lora_model_parameters.txt")
        
        lora_params = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
        print(f"\nFound {len(lora_params)} LoRA parameters")
        print("\nSample LoRA parameters:")
        for param in lora_params[:5]:
            print(f"- {param}")

    print("Model loaded successfully")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        print("Token embeddings resized")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return tokenizer, model, image_processor

def process_adam_optimizer_states(args, model):
    """Process and combine Adam optimizer states from DeepSpeed checkpoints."""
    # Find the global_step directory
    global_step_dirs = [d for d in os.listdir(args.model_path) if d.startswith("global_step")]
    if not global_step_dirs:
        raise ValueError(f"No global_step directory found in {args.model_path}")
        
    global_step_dir = os.path.join(args.model_path, global_step_dirs[0])
    optim_files = sorted(glob.glob(os.path.join(global_step_dir, "*optim_states.pt")))
    print(f"Found {len(optim_files)} optimizer state files")
    
    # Load and process optimizer states
    adam_optimizer_state = load_optimizer_states(optim_files)
    
    # Map parameters to optimizer states
    named_optimizer_states = map_parameters_to_states(model, adam_optimizer_state, len(optim_files))
    
    return named_optimizer_states

def load_optimizer_states(optim_files):
    """Load and combine optimizer states from multiple files."""
    adam_optimizer_state = {}
    for i, optim_file in enumerate(optim_files):
        print(f"Loading file {i}: {os.path.basename(optim_file)}")
        state_dict = torch.load(optim_file, map_location="cpu")
        optimizer_states = state_dict["optimizer_state_dict"]["optimizer_state_dict"]
        
        offset = i * 1000
        states = optimizer_states["state"]
        for param_id, state in states.items():
            if "exp_avg" in state and "exp_avg_sq" in state:
                new_param_id = param_id + offset
                adam_optimizer_state[new_param_id] = {
                    "exp_avg": state["exp_avg"].clone(),
                    "exp_avg_sq": state["exp_avg_sq"].clone()
                }
        
        del state_dict, optimizer_states
        torch.cuda.empty_cache()
        print(f"File {i}: Added {len(states)} states")
    
    print_optimizer_stats(adam_optimizer_state)
    return adam_optimizer_state

def map_parameters_to_states(model, adam_optimizer_state, num_ranks):
    """Map model parameters to optimizer states."""
    trainable_params = get_trainable_params_map(model)
    combined_states = combine_optimizer_states(adam_optimizer_state, num_ranks)
    return map_states_to_params(trainable_params, combined_states)

# Helper functions
def print_optimizer_stats(adam_optimizer_state):
    """Print statistics about optimizer states."""
    print(f"Total parameters with Adam states: {len(adam_optimizer_state)}")
    total_elements = sum(state['exp_avg'].numel() for state in adam_optimizer_state.values())
    print(f"Total elements: {total_elements} ({total_elements/1e9:.2f}B parameters)")

def get_trainable_params_map(model):
    """Create mapping of trainable parameters."""
    trainable_params = {}
    current_pos = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Skipping frozen parameter: {name}")
            continue
        param_size = param.numel()
        trainable_params[name] = (current_pos, param_size, param.shape)
        current_pos += param_size
    print(f"Total trainable parameters mapped: {current_pos}")
    return trainable_params

def combine_optimizer_states(adam_optimizer_state, num_ranks):
    """Combine optimizer states from multiple ranks."""
    combined_exp_avg = []
    combined_exp_avg_sq = []
    for i in range(num_ranks):
        base_id = i * 1000
        for state_id in [base_id, base_id + 1]:
            if state_id in adam_optimizer_state:
                combined_exp_avg.append(adam_optimizer_state[state_id]['exp_avg'])
                combined_exp_avg_sq.append(adam_optimizer_state[state_id]['exp_avg_sq'])
    return {
        'exp_avg': torch.cat(combined_exp_avg),
        'exp_avg_sq': torch.cat(combined_exp_avg_sq)
    }

def map_states_to_params(trainable_params, combined_states):
    """Map combined states back to parameter names."""
    named_optimizer_states = {}
    for name, (start_pos, size, shape) in trainable_params.items():
        try:
            named_optimizer_states[name] = {
                "exp_avg": combined_states['exp_avg'][start_pos:start_pos + size].view(shape),
                "exp_avg_sq": combined_states['exp_avg_sq'][start_pos:start_pos + size].view(shape)
            }
        except Exception as e:
            print(f"Error mapping parameter {name}: {e}")
            print(f"Expected shape: {shape}, size: {size}, start_pos: {start_pos}")
            raise
    
    print(f"Mapped {len(named_optimizer_states)} parameters")
    print(f"Sample parameter names: {list(named_optimizer_states.keys())[:5]}")
    return named_optimizer_states


def collect_info(args, dataloader, model):
    """Collect gradients, representations, or deltas based on info_type."""
    
    # Initialize gradient collector
    collector = GradientCollector(
        model=model,
        output_dir=args.output_path,
        proj_dim=args.gradient_projection_dimension,
        gradient_type=args.gradient_type,
        adam_state=process_adam_optimizer_states(args, model) if args.gradient_type == "adam" else None,
        save_interval=160
    )
    
    # Map info_type to collection method
    collection_methods = {
        "reps": collector.collect_reps,
        "grads": collector.collect_grads,
        "delta": collector.collect_delta_grads
    }
    
    if args.info_type not in collection_methods:
        raise ValueError(f"Unknown info_type: {args.info_type}")
        
    print(f"Collecting {args.info_type}")
    collection_methods[args.info_type](dataloader, max_samples=args.max_samples)

def set_lora_trainable(model):
    """Set requires_grad=True for LoRA parameters."""
    lora_params_found = False
    trainable_count = 0
    
    # Set requires_grad for LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params_found = True
            trainable_count += 1
            print(f"Set trainable: {name}")
    
    if not lora_params_found:
        print("\n⚠️ WARNING: No LoRA parameters found in the model!")
    else:
        print(f"\nSet {trainable_count} LoRA parameters to trainable")
    
    model.print_trainable_parameters()
    return lora_params_found

def main():
    parser = HfArgumentParser((SelectionArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    print("Command-line arguments:", args)
    assert args.task is not None or args.train_file is not None
    tokenizer, model, image_processor = initialize_model(args)
    data_args.image_processor = image_processor
    set_lora_trainable(model)
    model.print_trainable_parameters()  
    print(args.train_file)
    dataset = get_dataset(data_args, args.train_file, tokenizer, args.max_length)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
    print(f"Dataset loaded from {'task' if args.task else 'file'}:", args.task or args.train_file)
    print("Number of samples in the dataset:", len(dataset))
    collect_info(args, dataloader, model)
    print("Operation completed successfully")

if __name__ == "__main__":
    main()


