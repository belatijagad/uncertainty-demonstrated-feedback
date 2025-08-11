import unittest
import tempfile
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig

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