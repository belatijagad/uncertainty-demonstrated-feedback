from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.trainers import DPOTrainer
from alignment.utils import batched_generate

from vllm import LLM
from vllm.lora.request import LoRARequest

class DITTOTrainer(DPOTrainer):
    def __init__(self, model: PreTrainedModel, adapter_name: str, config: DictConfig, 
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, optimizer: Optimizer, device: str,
                 eval_dataloader: Optional[DataLoader] = None, callbacks: Optional[list[TrainerCallback]] = None, 
                 vllm_model: LLM = None, lora_save_path: Optional[Path] = None):
        super().__init__(model, adapter_name, config, tokenizer, train_dataloader, optimizer, device, eval_dataloader, callbacks, lora_save_path)
        self.llm = vllm_model

    def _generate_samples(self) -> tuple[list[str], list[str]]:
        """Generate samples from policy for wandb logging using KeyDataset."""        
        self.model.eval()

        eval_dataset = self.eval_dataloader.dataset
        prompts = [example["prompt"] for example in eval_dataset]
        
        lora_request = None
        lora_request = (
            None if self.llm is None
            else LoRARequest(self.adapter_name, 1, str(self.lora_save_path / "ditto"))
            )

        gen_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.config.max_length - self.config.max_prompt_len,
        }

        generations = batched_generate(
            prompts,
            model=self.llm if self.llm is not None else self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            lora_request=lora_request,
            disable_peft_adapter=False,
            adapter_name=self.adapter_name,
            gen_kwargs=gen_kwargs,
        )
        all_policy_samples = [generation["generated_text"] for generation in generations]

        self.model.train()
                
        return prompts, all_policy_samples
