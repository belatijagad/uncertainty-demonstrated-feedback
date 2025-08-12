import os
import unittest
import tempfile

from omegaconf import DictConfig

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from alignment.trainers import DPOTrainer
from alignment.collators import OfflineDPODataCollator, DITTODataCollator
from alignment.callbacks import ResampleCallback

class DPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.policy = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataset = Dataset.from_dict({
            "prompt": ["What is AI?", "How to code?"],
            "chosen": ["AI is artificial intelligence.", "Use programming languages."],
            "rejected": ["I don't know.", "Just type stuff."]
        })
        
        self.config = DictConfig({
            "epochs": 1,
            "beta": 0.1,
            "warmup_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 1,
            "eval_steps": 5,
            "save_steps": 100,
            "sample_during_eval": False,
            "gradient_accumulation_steps": 1,
            "max_length": 64,
            "push_to_hub": False,
            "repo_id": "test/repo"
        })

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def test_dpo_training_runs(self):
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(self.policy.parameters(), lr=1e-5)
        trainer = DPOTrainer(
            policy=self.policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        initial_param = next(self.policy.parameters()).clone()
        trainer.train()
        
        final_param = next(self.policy.parameters())
        self.assertFalse(torch.allclose(initial_param, final_param, rtol=1e-6))
    
    def test_dpo_evaluation_runs(self):
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(self.policy.parameters(), lr=1e-5)
        trainer = DPOTrainer(
            policy=self.policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        metrics = trainer.evaluate()
        
        self.assertIsInstance(metrics, dict)
        expected_metrics = [
            "eval_loss",
            "eval_rewards/chosen", 
            "eval_rewards/rejected",
            "eval_rewards/accuracies",
            "eval_rewards/margins"
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing expected metric: {metric}")
    
    def test_dpo_save_loads(self):
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(self.policy.parameters(), lr=1e-5)
        trainer = DPOTrainer(
            policy=self.policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.save(output_dir=tmp_dir)
            expected_files = [
                "config.json",
                "tokenizer.json", 
                "tokenizer_config.json",
                "optimizer.pt",
                "scheduler.pt"
            ]
            for file_name in expected_files:
                file_path = os.path.join(tmp_dir, file_name)
                if file_name in ["optimizer.pt", "scheduler.pt"]:
                    self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")

    def test_lora_training_runs(self):
        lora_policy = get_peft_model(self.policy, self.lora_config, adapter_name="dpo")
        lora_policy.set_adapter("dpo")
        
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(lora_policy.parameters(), lr=1e-4)  # Higher LR for LoRA
        
        trainer = DPOTrainer(
            policy=lora_policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        trainable_param_names = [name for name, param in lora_policy.named_parameters() if param.requires_grad]
        lora_param_names = [name for name in trainable_param_names if 'lora_' in name]
        
        self.assertGreater(len(lora_param_names), 0, "No LoRA parameters found")
        
        initial_lora_params = {}
        for name, param in lora_policy.named_parameters():
            if param.requires_grad and 'lora_' in name:
                initial_lora_params[name] = param.clone().detach()
        
        trainer.train()
        
        changed_params = 0
        for name, param in lora_policy.named_parameters():
            if name in initial_lora_params:
                if not torch.allclose(initial_lora_params[name], param, rtol=1e-6):
                    changed_params += 1
        
        self.assertGreater(changed_params, 0, "No LoRA parameters were updated during training")
    
    def test_lora_evaluation_runs(self):
        lora_policy = get_peft_model(self.policy, self.lora_config, adapter_name="dpo")
        lora_policy.set_adapter("dpo")
        
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(lora_policy.parameters(), lr=1e-4)
        
        trainer = DPOTrainer(
            policy=lora_policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        metrics = trainer.evaluate()
        
        self.assertIsInstance(metrics, dict)
        expected_metrics = [
            "eval_loss",
            "eval_rewards/chosen", 
            "eval_rewards/rejected",
            "eval_rewards/accuracies",
            "eval_rewards/margins"
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing expected metric: {metric}")

    def test_lora_save_and_load(self):
        lora_policy = get_peft_model(self.policy, self.lora_config, adapter_name="dpo")
        lora_policy.set_adapter("dpo")
        
        collator = OfflineDPODataCollator(self.tokenizer, max_length=64, max_prompt_length=32)
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(lora_policy.parameters(), lr=1e-4)
        
        trainer = DPOTrainer(
            policy=lora_policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            optimizer=optimizer,
            wandb_run=None
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.save(output_dir=tmp_dir)
            
            expected_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "optimizer.pt",
                "scheduler.pt"
            ]
            
            for file_name in expected_files:
                file_path = os.path.join(tmp_dir,file_name)
                if file_name in [f"{lora_policy.active_adapter}/adapter_config.json"]:
                    self.assertTrue(os.path.exists(file_path), f"Missing LoRA file: {file_name}")
                elif file_name in ["optimizer.pt", "scheduler.pt"]:
                    self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")

class DITTOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.policy = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataset = Dataset.from_dict({
            "prompt": ["What is AI?", "How to code?", "Explain Python"],
            "chosen": ["AI is artificial intelligence.", "Use programming languages.", "Python is a programming language."],
            "rejected": ["I don't know.", "Just type stuff.", "It's some language."],
            "author": ["author1", "author1", "author2"]  # Required for DITTO
        })
        
        self.config = DictConfig({
            "epochs": 1,
            "beta": 0.1,
            "warmup_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 1,
            "eval_steps": 5,
            "save_steps": 100,
            "sample_during_eval": False,
            "gradient_accumulation_steps": 1,
            "max_length": 64,
            "push_to_hub": False,
            "repo_id": "test/repo"
        })
        
        self.resample_config = {
            "frac_expert": 0.7,
            "frac_replay": 0.2,
            "frac_noisy": 0.1,
            "rescale_batch": 1,
            "bootstrap_count": 2, 
            "batch_size": 2
        }

    def test_ditto_training(self):
        """Test DITTO training with resampling callback."""
        collator = DITTODataCollator(
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            max_length=64,
            max_prompt_length=32,
            **self.resample_config
        )
        collator.model = self.policy
        
        dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collator)
        
        optimizer = AdamW(self.policy.parameters(), lr=1e-5)
        
        # Add ResampleCallback for DITTO
        callbacks = [ResampleCallback(collator=collator, model=self.policy, resample_rate=1)]
        
        trainer = DPOTrainer(
            policy=self.policy,
            ref_policy=self.ref_policy,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataloader=dataloader,
            optimizer=optimizer,
            callbacks=callbacks,
            wandb_run=None
        )
        
        initial_param = next(self.policy.parameters()).clone()
        
        trainer.train()
        
        final_param = next(self.policy.parameters())
        self.assertFalse(torch.allclose(initial_param, final_param, rtol=1e-6))

if __name__ == "__main__":
    unittest.main()