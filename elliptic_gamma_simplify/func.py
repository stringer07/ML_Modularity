from torch.utils.data import DataLoader, Sampler
import torch
import random
from transformers import Trainer
import os


class DynamicBatchSampler(Sampler):
    
    def _calculate_actual_num_batches(self):
        num_batches = 0
        current_batch_tokens = 0
        current_batch_size = 0
        
        for bucket in self.buckets.values():
            for idx in bucket:
                sample_len = self.lengths[idx]
                
                if (current_batch_tokens + sample_len > self.max_tokens_per_batch or 
                    current_batch_size >= self.max_batch_size):
                    num_batches += 1
                    current_batch_tokens = sample_len
                    current_batch_size = 1
                else:
                    current_batch_tokens += sample_len
                    current_batch_size += 1
                    
        if current_batch_size > 0:
            num_batches += 1
            
        return num_batches
    
    def __init__(self, dataset, max_tokens_per_batch, shuffle=True, max_batch_size = 32, length_bucket_size= 64):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.max_batch_size = max_batch_size
        self.length_bucket_size = length_bucket_size  
        
        def get_length(example):
            return {
                'length': max(
                    len(example['input_ids']), 
                    len(example['labels'])
                )
            }
        
        print("Calculating sample lengths...")
        self.dataset = self.dataset.map(
            get_length,
            num_proc = 10,  
            desc="Calculating lengths",
            load_from_cache_file=True
        )
        
        self.lengths = list(self.dataset['length'])
        self.buckets = self._create_length_buckets()
        print(f"Created {len(self.buckets)} length buckets")
        
        self.total_samples = len(self.dataset)
        print(f"Total samples: {self.total_samples}")
        
        self.avg_tokens_per_sample = sum(self.lengths) / self.total_samples
        print(f"Avg tokens per sample: {self.avg_tokens_per_sample:.2f}")
        
        self.estimated_batch_size = self.max_tokens_per_batch / self.avg_tokens_per_sample
        print(f"Estimated batch size: {self.estimated_batch_size:.2f}")
        
        self.num_batches = self._calculate_actual_num_batches()
        print(f"Number of batches: {self.num_batches}")
        
    def _create_length_buckets(self):

        buckets = {}
        for idx, length in enumerate(self.lengths):

            bucket_idx = length // self.length_bucket_size
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(idx)
        return buckets
    
    def __iter__(self):
        bucket_indices = list(self.buckets.keys())
        
        # if self.shuffle:
        #     random.shuffle(bucket_indices)

#===============================================================================#
        bucket_indices.sort(reverse=True) 
        print(f"!!! DEBUG MODE: Processing buckets in descending order (Largest first) !!!")
    
#==========================================================================#
        for bucket_idx in bucket_indices:
            bucket = self.buckets[bucket_idx]
            if self.shuffle:
                random.shuffle(bucket)
            
            batch = []
            current_batch_tokens = 0
            
            for idx in bucket:
                sample_len = self.lengths[idx]
                
                if (current_batch_tokens + sample_len > self.max_tokens_per_batch or 
                    len(batch) >= self.max_batch_size) and batch:
                    yield batch
                    batch = []
                    current_batch_tokens = 0
                
                batch.append(idx)
                current_batch_tokens += sample_len
            
            if batch:
                yield batch
    
    def __len__(self):
        return self.num_batches


class CustomTrainer(Trainer):
    def __init__(self, max_tokens_per_batch, max_memory_usage=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.max_tokens_per_batch = max_tokens_per_batch
        
        self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  
        self.max_memory = self.total_memory * max_memory_usage
        

    def get_train_dataloader(self):
        train_batch_sampler = DynamicBatchSampler(
            dataset=self.train_dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
            shuffle=True
        )
 

        return DataLoader(
            self.train_dataset,       
            batch_sampler=train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_batch_sampler = DynamicBatchSampler(
            dataset=eval_dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
            shuffle=False
        )
        return DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor
        )