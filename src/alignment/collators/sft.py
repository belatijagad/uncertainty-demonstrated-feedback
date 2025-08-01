from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass

@dataclass
class SFTDataCollator:
    """Data collator that applies the tokenizer's chat template for SFT."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(self, features: list[dict]) -> dict:
        formatted_texts = [self.tokenizer.apply_chat_template(f["messages"], tokenize=False, add_generation_prompt=False) for f in features]
        
        batch = self.tokenizer(
            formatted_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        batch["labels"] = batch["input_ids"].clone()
        for i, feature in enumerate(features):
            prompt_turns = feature["messages"][:-1]
            prompt_str = self.tokenizer.apply_chat_template(prompt_turns, tokenize=False, add_generation_prompt=True)
            prompt_len = len(self.tokenizer(prompt_str, add_special_tokens=True).input_ids)
            
            batch["labels"][i, :prompt_len] = -100

        batch["labels"][batch["attention_mask"] == 0] = -100
        
        return batch
