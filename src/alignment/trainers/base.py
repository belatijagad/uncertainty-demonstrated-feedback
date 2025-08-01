import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from tqdm import tqdm
from typing import Optional, Literal
from collections import defaultdict
from omegaconf import DictConfig
from alignment.callbacks.callbacks import TrainerCallback

from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model: PreTrainedModel, config: DictConfig,
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader,
                 optimizer: Optimizer, callbacks: Optional[list[TrainerCallback]] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.callbacks = callbacks if callbacks is not None else []
        
        self.device= 'cuda' if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.scheduler.warmup_steps,
            num_training_steps=num_train_steps
        )
        self.step_counter = 0
        
    @abstractmethod
    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.tensor, dict]:
        """
        Computes the loss and a dictionary of metrics for a given batch
        """
        pass
    
    def train(self):
        """Main training loop."""
        print(f"Starting {self.__class__.__name__} training...")
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        pbar = tqdm(total=num_train_steps, desc="Training Steps")
        data_iterator = iter(self.train_dataloader)
        
        grad_acc_steps = self.config.trainer.get("gradient_accumulation_steps", 1)

        for cb in self.callbacks: cb.on_train_begin(trainer=self)
        self.optimizer.zero_grad()

        for i in range(num_train_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss, metrics = self._get_batch_metrics(batch, "train")
            loss = loss / grad_acc_steps
            loss.backward()

            if (i + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step_counter += 1

                pbar.update(grad_acc_steps)
                pbar.set_postfix({"loss": metrics.get("loss", 0.0)})
                
                for cb in self.callbacks:
                    cb.on_step_end(step=self.step_counter, metrics=metrics, trainer=self)
        
        for cb in self.callbacks: cb.on_train_end(trainer=self)
        pbar.close()
    
    def evaluate(self) -> dict:
        """Main evaluation loop."""
        print(f"Running {self.__class__.__name__} evaluation...")
        self.model.eval()
        all_metrics = defaultdict(list)

        with torch.inference_mode():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                _, metrics = self._get_batch_metrics(batch, "eval")
                for key, value in metrics.items():
                    all_metrics[key].append(value)

        final_metrics = {key: torch.tensor(values).mean().item() for key, values in all_metrics.items()}
        
        self.model.train()
        print(f"Evaluation finished: {final_metrics}")
        return final_metrics

    def save(self):
        """Saves the model, optimizer, and scheduler states."""
        output_dir = self.config.model.output_path
        print(f"Saving model and training states to {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.optimizer.state_dict(), f"{output_dir}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{output_dir}/scheduler.pt")
        
    @staticmethod
    def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int = -100) -> torch.FloatTensor:
        """Computes the log probabilities of the given labels under the given logits."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id)

        # Set pad token labels to 0 to avoid index errors; the loss is masked anyway
        labels[labels == label_pad_token_id] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * loss_mask).sum(-1)
    