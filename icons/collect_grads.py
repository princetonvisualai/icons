"""
    This script is used for collecting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""
import os
from enum import Enum
from typing import List, Optional

import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType


def get_trak_projector(device: torch.device):
    """Get trak projectors"""
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        return CudaProjector
    except:
        return BasicProjector

class GradientCollector:
    def __init__(
        self, 
        model,
        output_dir: str,
        proj_dim: List[int],
        gradient_type: str = "sgd",  # "sgd" or "adam"
        adam_state: Optional[dict] = None,
        project_interval: int = 16,
        save_interval: int = 160
    ):
        self.model = model
        self.output_dir = output_dir
        self.proj_dim = proj_dim
        self.gradient_type = gradient_type
        self.project_interval = project_interval
        self.save_interval = save_interval
        
        # Setup device and projectors
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Setup Adam state if needed
        if gradient_type == "adam":
            assert adam_state is not None, "Adam state required for adam gradient type"
            self.m_state = adam_state['m']
            self.v_state = adam_state['v']
            
        self.setup_projectors()
            
    def setup_projectors(self):
        """Initialize projectors for each dimension"""
        projector_cls = get_trak_projector(self.device)
        self.n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {self.n_trainable_params}")
        
        # Add verification
        for dim in self.proj_dim:
            if dim >= self.n_trainable_params:
                raise ValueError(f"Projection dimension ({dim}) must be smaller than number of parameters ({self.n_trainable_params})")
        
        self.projectors = []
        self.output_dirs = {}
        
        for dim in self.proj_dim:
            proj = projector_cls(
                grad_dim=self.n_trainable_params,
                proj_dim=dim,
                seed=0,
                proj_type=ProjectionType.rademacher,
                device=self.device,
                dtype=self.dtype,
                block_size=128,
                max_batch_size=16
            )
            self.projectors.append(proj)
            
            # Setup output directory
            output_dir = os.path.join(self.output_dir, f"dim{dim}")
            self.output_dirs[dim] = output_dir
            os.makedirs(output_dir, exist_ok=True)


    def obtain_gradients(self, batch, remove_image: bool = False) -> torch.Tensor:
        """Compute gradients with option to remove or replace image with Gaussian noise"""
        # Add safety check for sequence length
        max_length = 2048  # Set this to your model's maximum length
        if batch["input_ids"].size(1) > max_length:
            print(f"Warning: Truncating sequence from {batch['input_ids'].size(1)} to {max_length}")
            batch = {
                "input_ids": batch["input_ids"][:, :max_length],
                "attention_mask": batch["attention_mask"][:, :max_length],
                "labels": batch["labels"][:, :max_length],
            }
            if "image" in batch:
                batch["image"] = batch["image"]
        
        # Clear cache before processing
        torch.cuda.empty_cache()
        
        try:
            # Move all inputs to the correct device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            images = batch.get("image", None)

            if images is not None:
                if not remove_image:
                    images = images.to(dtype=self.dtype, device=self.device)
                else:
                    # Replace images with Gaussian noise
                    noise = torch.randn_like(images, dtype=self.dtype, device=self.device)
                    images = noise

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs.loss

            # Free up memory before computing gradients
            del outputs
            torch.cuda.empty_cache()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_layers = torch.autograd.grad(
                loss, 
                trainable_params, 
                retain_graph=True,
                allow_unused=True
            )
            
            # Handle None gradients (unused parameters)
            grad_layers = [
                torch.zeros_like(param) if grad is None else grad
                for grad, param in zip(grad_layers, trainable_params)
            ]
            
            if self.gradient_type == "adam":
                grad_layers = self.apply_adam_update(grad_layers)
            
            # Clear more memory
            del loss
            torch.cuda.empty_cache()
            
            return torch.cat([grad.flatten() for grad in grad_layers])
        
        except torch.cuda.OutOfMemoryError:
            # If we still get OOM, try with an even smaller sequence length
            torch.cuda.empty_cache()
            if max_length > 512:
                print(f"OOM error, retrying with sequence length {max_length//2}")
                batch = {
                    "input_ids": batch["input_ids"][:, :max_length//2],
                    "attention_mask": batch["attention_mask"][:, :max_length//2],
                    "labels": batch["labels"][:, :max_length//2],
                }
                if "image" in batch:
                    batch["image"] = batch["image"]
                return self.obtain_gradients(batch, remove_image)
            else:
                raise  # If we're already at a very small sequence length, raise the error

    def obtain_gradients_old(self, batch, remove_image: bool = False) -> torch.Tensor:
        """Compute gradients with option to remove or replace image with Gaussian noise"""
        # Move all inputs to the correct device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        images = batch.get("image", None)

        if images is not None:
            if not remove_image:
                images = images.to(dtype=self.dtype, device=self.device)
            else:
                # Replace images with Gaussian noise
                noise = torch.randn_like(images, dtype=self.dtype, device=self.device)
                images = noise

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_layers = torch.autograd.grad(
            loss, 
            trainable_params, 
            retain_graph=True,
            allow_unused=True
        )
        
        # Handle None gradients (unused parameters)
        grad_layers = [
            torch.zeros_like(param) if grad is None else grad
            for grad, param in zip(grad_layers, trainable_params)
        ]
        
        if self.gradient_type == "adam":
            grad_layers = self.apply_adam_update(grad_layers)
        
        
        return torch.cat([grad.flatten() for grad in grad_layers])
    
    
    def apply_adam_update(self, grads):
        """Apply Adam optimizer update to gradients"""
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        updated_grads = []
        for i, g in enumerate(grads):
            # Update biased first moment estimate
            self.m_state[i] = beta1 * self.m_state[i] + (1 - beta1) * g
            # Update biased second raw moment estimate
            self.v_state[i] = beta2 * self.v_state[i] + (1 - beta2) * g * g
            
            # Compute bias-corrected estimates
            m_hat = self.m_state[i] / (1 - beta1)
            v_hat = self.v_state[i] / (1 - beta2)
            
            # Compute Adam update
            updated_grad = m_hat / (torch.sqrt(v_hat) + eps)
            updated_grads.append(updated_grad)
            
        return updated_grads

    def _project(self, full_grads, projected_grads):
        """Project gradients efficiently in chunks"""
        # Convert to half precision to save memory
        current_grads = torch.stack(full_grads).to(torch.float16)
        
        # Add verification
        if current_grads.dim() != 2 or current_grads.size(1) != self.n_trainable_params:
            raise ValueError(f"Unexpected gradient shape: {current_grads.shape}, expected (batch_size, {self.n_trainable_params})")
        
        # Project all at once for speed
        for i, projector in enumerate(self.projectors):
            projected = projector.project(current_grads.cuda(), model_id=0)
            projected_grads[self.proj_dim[i]].extend(projected.cpu().chunk(current_grads.size(0)))
        
        # Clear memory
        del current_grads
        torch.cuda.empty_cache()

    def _save(self, projected_grads, sample_count):
        """Save projected gradients with sample count in filename"""
        for dim in self.proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])
            # Use sample_count in filename
            outfile = os.path.join(self.output_dirs[dim], f"grads-{sample_count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(f"Saving {outfile}, shape: {projected_grads[dim].shape}, samples processed: {sample_count}")
            projected_grads[dim] = []  # Clear after saving

    def collect_reps(self, dataloader: DataLoader, max_samples: Optional[int] = None):
        """Collect model representations"""
        full_reps = []
        projected_reps = {dim: [] for dim in self.proj_dim}
        count = 0

        for idx, batch in enumerate(tqdm(dataloader)):
            if max_samples and idx >= max_samples:
                break
                
            count += 1
            outputs = self.model(**batch)
            reps = outputs.hidden_states[-1][:, 0].flatten()  # Use CLS token
            full_reps.append(reps)
            
            # Project and save periodically
            if idx % self.project_interval == 0:
                self._project(full_reps, projected_reps)
                full_reps = []
                
            if idx % self.save_interval == 0:
                self._save(projected_reps, count)
                
        # Final projection and save
        if full_reps:
            self._project(full_reps, projected_reps)
            self._save(projected_reps, count)
            
        self.merge_results()

    def collect_grads(self, dataloader: DataLoader, max_samples: Optional[int] = None):
        """Collect regular gradients with efficient memory-speed tradeoff"""
        full_grads = []
        projected_grads = {dim: [] for dim in self.proj_dim}
        total_processed = 0
        chunk_size = 4  # Larger chunk size for less frequent saving
        
        print(f"Total samples to process: {len(dataloader.dataset)}")
        
        for idx, batch in enumerate(tqdm(dataloader)):
            if max_samples and total_processed >= max_samples:
                break
            
            # XINDI: temporarily use efficient version to use 3090 gpus
            grads = self.obtain_gradients(batch)
            # grads = self.obtain_gradients_efficient(batch)
            full_grads.append(grads)
            total_processed += 1
            
            # Project when chunk is full or at the end
            if len(full_grads) >= chunk_size or idx == len(dataloader) - 1:
                print(f"Processing chunk of size {len(full_grads)}")
                self._project(full_grads, projected_grads)
                self._save(projected_grads, total_processed)  # Pass total_processed instead of count
                full_grads = []  # Clear accumulated grads
                
            self.model.zero_grad()
        
        print(f"Total samples processed: {total_processed}")
        self.merge_results()

    def collect_delta_grads(self, dataloader: DataLoader, max_samples: Optional[int] = None):
        """Collect delta gradients (difference between with and without image)"""
        full_delta_grads = []
        projected_delta_grads = {dim: [] for dim in self.proj_dim}
        count = 0
        chunk_size = 8  # Adjust this based on your GPU memory and speed requirements

        for idx, batch in enumerate(tqdm(dataloader)):
            if max_samples and idx >= max_samples:
                break
                
            count += 1
            grads_without_image = self.obtain_gradients(batch, remove_image=True)
            full_delta_grads.append(grads_without_image)
            
            # Project when chunk is full
            if len(full_delta_grads) >= chunk_size:
                self._project(full_delta_grads, projected_delta_grads)
                full_delta_grads = []  # Clear accumulated grads
                
            self.model.zero_grad()
                
            # # Save periodically and ensure we don't lose data
            # if ((idx + 1) % self.save_interval == 0) or (idx == len(dataloader) - 1):
            #     self._save(projected_delta_grads, count)

            # # Save periodically

            if (idx + 1) % self.save_interval == 0:
                self._save(projected_delta_grads, count)
                
        # Handle remaining grads
        if full_delta_grads:
            self._project(full_delta_grads, projected_delta_grads)
            self._save(projected_delta_grads, count)
                
        self.merge_results()

    def merge_results(self):
        """Merge and save final results"""
        for dim in self.proj_dim:
            output_dir = self.output_dirs[dim]
            merge_info(output_dir, prefix="grads", normalize_data=True)
            merge_info(output_dir, prefix="grads", normalize_data=False)

    def verify_projection(self, original, projected, proj_dim):
        """Verify projection properties"""
        # Check output dimension
        if projected.size(1) != proj_dim:
            raise ValueError(f"Wrong projection dimension: {projected.size(1)}, expected {proj_dim}")
        
        # Check preservation of relative distances (optional)
        if original.size(0) > 1:
            orig_dist = torch.pdist(original)
            proj_dist = torch.pdist(projected)
            correlation = torch.corrcoef(torch.stack([orig_dist, proj_dist]))[0,1]
            print(f"Distance preservation correlation: {correlation:.3f}")

    def obtain_gradients_efficient(self, batch):
        """Memory-efficient version of gradient computation"""
        # Move all inputs to the correct device
        torch.cuda.empty_cache()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        images = batch.get("image", None)

        if images is not None:
            images = images.to(dtype=self.dtype, device=self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss
        loss.backward()

        # Process gradients in chunks
        grad_layers = []
        for param in self.model.parameters():
            if param.requires_grad:
                grad = param.grad
                if grad is not None:
                    # Process in smaller chunks
                    chunk_size = 1000000  # Adjust this value based on your GPU memory
                    flattened = grad.flatten()
                    chunks = torch.split(flattened, chunk_size)
                    grad_layers.extend(chunks)
        
        # Concatenate chunks carefully
        final_grad = []
        for chunk in grad_layers:
            final_grad.append(chunk.cpu())  # Move to CPU immediately
        
        # Clear memory
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        return torch.cat(final_grad).cuda()  # Only bring back to GPU when needed

def merge_info(output_dir: str, prefix="grads", normalize_data: bool = False):
    """Merge files with optional normalization"""
    info = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    
    merged_data = []
    for file in tqdm(info, desc=f"Merging {'and normalizing ' if normalize_data else ''}files"):
        data = torch.load(os.path.join(output_dir, file))
        if normalize_data:
            data = normalize(data, dim=1)
        merged_data.append(data)

    if merged_data:
        merged_data = torch.cat(merged_data, dim=0)
        suffix = "normalized" if normalize_data else "unormalized"
        output_file = os.path.join(output_dir, f"all_{suffix}.pt")
        torch.save(merged_data, output_file)
