import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.trainers import DPOTrainer
from alignment.utils import batched_generate

try:
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

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
        
        max_prompt_len = self.config.get("max_prompt_length", self.config.max_length // 2)

        lora_request = (
            None if self.lora_adapter_path is None or not VLLM_AVAILABLE
            else LoRARequest("ditto", 1, str(Path("../temp/ditto").resolve()))
            )
        generations = batched_generate(
            prompts,
            max_new_tokens=self.config.max_length - max_prompt_len,
            model=self.llm if self.llm is not None else self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            lora_request=lora_request,
            num_return_sequences=1,
            do_sample=True,
            disable_peft_adapter=False,
            adapter_name="ditto",
        )
        all_policy_samples = [texts[0] for texts in generations]

        self.model.train()
                
        return prompts, all_policy_samples
