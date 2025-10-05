import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.pipelines import pipeline
from transformers import PreTrainedTokenizer, PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.trainers.dpo import DPOTrainer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


logger = logging.getLogger(__name__)

class DITTOTrainer(DPOTrainer):
    def __init__(self, model: PreTrainedModel, adapter_name: str, config: DictConfig, 
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, optimizer: Optimizer, device: str,
                 eval_dataloader: Optional[DataLoader] = None, callbacks: Optional[list[TrainerCallback]] = None, vllm_model: LLM = None):
        super().__init__(model, adapter_name, config, tokenizer, train_dataloader, optimizer, device, eval_dataloader, callbacks)
        self.llm = vllm_model

    def _generate_samples(self) -> tuple[list[str], list[str]]:
        """Generate samples from policy for wandb logging using KeyDataset."""        
        self.model.eval()

        eval_dataset = self.eval_dataloader.dataset
        prompts = [example["prompt"] for example in eval_dataset]
        all_policy_samples = []
        
        max_prompt_len = self.config.get("max_prompt_length", self.config.max_length // 2)

        if self.config.get("use_vllm"):
            torch.cuda.empty_cache()
            self.llm.wake_up()
            sampling_params = SamplingParams(
                max_tokens=self.config.max_length-max_prompt_len,
            )
            adapter_path = str(Path("../temp/ditto").resolve())
            outputs = self.llm.generate(
                prompts, sampling_params, 
                lora_request=LoRARequest("ditto", 1, adapter_path))
            self.llm.sleep(level=2)
            all_policy_samples.extend(outputs)
        else:
            gen_kwargs = {
                "max_new_tokens": self.config.max_length-max_prompt_len,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "truncation": True,
            }

            policy_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                device=self.device
            )
            
            with torch.inference_mode():
                for prompt in tqdm(prompts, desc="Generating policy samples", leave=False):
                    policy_out = policy_generator(prompt, **gen_kwargs)
                    all_policy_samples.append(policy_out[0]["generated_text"])
                        
        self.model.train()
                
        return prompts, all_policy_samples
