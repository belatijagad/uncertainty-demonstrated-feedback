import shutil
import tempfile
import unittest

import torch
from vllm import LLM
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.utils import batched_generate

class TestBatchedGenerateHuggingFace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.model.eval()

    def test_single_prompt_returns_logprobs(self):
        prompts = ["Who is Hoshimachi Suisei?"]
        gen_kwargs = {
            "num_return_sequences": 1,
            "max_new_tokens": 2,
            "return_dict_in_generate": True,
            "output_logits": True,
        }

        outputs = batched_generate(
            prompts,
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            gen_kwargs=gen_kwargs,
        )

        self.assertEqual(len(outputs), 1)
        sample = outputs[0]
        self.assertIn("generated_text", sample)
        self.assertIn("generated_token_ids", sample)
        self.assertIn("logprobs", sample)
        self.assertIsInstance(sample["logprobs"], torch.Tensor)
        self.assertGreaterEqual(sample["logprobs"].ndim, 2)
        self.assertGreater(sample["logprobs"].shape[-1], 0)


class TestBatchedGenerateVLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        cls.llm = LLM(
            model=cls.model_id,
            tokenizer=cls.model_id,
            max_model_len=128,
            enforce_eager=True,
            dtype="float16",
            enable_lora=True,
        )
        cls._lora_dir = tempfile.mkdtemp(prefix="vllm_lora_test_")
        cls._create_lora_adapter(cls._lora_dir)

    def test_vllm_single_prompt_structure(self):
        prompts = ["Who is Hoshimachi Suisei?"]
        gen_kwargs = {
            "num_return_sequences": 1,
        }

        outputs = batched_generate(
            prompts,
            model=self.llm,
            tokenizer=None,
            gen_kwargs=gen_kwargs,
        )

        self.assertEqual(len(outputs), 1)
        sample = outputs[0]
        self.assertIn("generated_text", sample)
        self.assertIn("generated_token_ids", sample)
        self.assertIn("logprobs", sample)
        self.assertGreater(len(sample["generated_token_ids"]), 0)
        print(f"Generated text: {sample['generated_text']}")

    def test_vllm_with_lora_request(self):
        prompts = ["What is your name?"]
        lora_request = LoRARequest("unit-test", 2, self._lora_dir)
        gen_kwargs = {
            "num_return_sequences": 1,
        }

        outputs = batched_generate(
            prompts,
            model=self.llm,
            tokenizer=None,
            lora_request=lora_request,
            gen_kwargs=gen_kwargs,
        )

        self.assertEqual(len(outputs), 1)
        sample = outputs[0]
        self.assertIn("generated_text", sample)
        self.assertGreater(len(sample["generated_token_ids"]), 0)
        print(f"Generated text: {sample['generated_text']}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_lora_dir") and cls._lora_dir:
            shutil.rmtree(cls._lora_dir, ignore_errors=True)

    @classmethod
    def _create_lora_adapter(cls, output_dir: str) -> None:
        base_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.save_pretrained(output_dir)


if __name__ == "__main__":
    unittest.main()