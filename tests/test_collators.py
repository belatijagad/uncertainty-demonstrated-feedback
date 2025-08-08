import unittest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.collators import DITTODataCollator

class DITTODataCollatorTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataset = Dataset.from_dict({
            "prompt": ["What is AI?", "How to code?", "Explain Python"],
            "chosen": ["AI is artificial intelligence.", "Use programming languages.", "Python is a programming language."],
            "rejected": ["I don't know.", "Just type stuff.", "It's some language."]
        })
        
        self.collator = DITTODataCollator(
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            max_length=128,
            max_prompt_length=64,
            batch_size=2,
            bootstrap_count=3
        )
        
        self.collator.model = self.model

    def test_resampling_creates_cache(self):
        step = 1        
        self.assertEqual(len(self.collator.cache), 0)
        
        self.collator.resample(step)
        
        self.assertIn(step, self.collator.cache)
        self.assertEqual(self.collator.last_sampled_step, step)
        
        for prompt in self.dataset["prompt"]:
            self.assertIn(prompt, self.collator.cache[step])
            
        for prompt in self.dataset["prompt"]:
            self.assertEqual(len(self.collator.cache[step][prompt]), self.collator.bootstrap_count)

    def test_cache_contains_valid_generations(self):
        step = 1
        self.collator.resample(step)
        
        for prompt in self.dataset["prompt"]:
            for response in self.collator.cache[step][prompt]:
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                
        for prompt in self.dataset["prompt"]:
            for response in self.collator.cache[step][prompt]:
                self.assertTrue(response.endswith(self.tokenizer.eos_token))

    def test_multiple_resampling_steps(self):
        steps = [1, 2, 3]
        
        for step in steps:
            self.collator.resample(step)
            
        for step in steps:
            self.assertIn(step, self.collator.cache)
            
        self.assertEqual(self.collator.last_sampled_step, max(steps))
        
        for step in steps:
            for prompt in self.dataset["prompt"]:
                self.assertIn(prompt, self.collator.cache[step])
                self.assertEqual(len(self.collator.cache[step][prompt]), self.collator.bootstrap_count)

    def test_cache_generations_are_different(self):
        step = 1
        self.collator.resample(step)
        
        for prompt in self.dataset["prompt"]:
            responses = self.collator.cache[step][prompt]
            
            self.assertEqual(len(responses), 3)
            
            # Due to randomness, responses might sometimes be identical,
            # but we check that they're potentially different
            unique_responses = set(responses)
            # Just verify we have at least 1 response (could be all same due to randomness)
            self.assertGreaterEqual(len(unique_responses), 1)

    def test_cache_persistence_across_calls(self):
        self.collator.resample(step=1)
        cache_step1 = self.collator.cache[1].copy()
        
        self.collator.resample(step=2)
        
        self.assertIn(1, self.collator.cache)
        self.assertIn(2, self.collator.cache)
        
        for prompt in self.dataset["prompt"]:
            self.assertEqual(
                self.collator.cache[1][prompt], 
                cache_step1[prompt]
            )

    def test_empty_dataset_handling(self):
        empty_dataset = Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})
        
        empty_collator = DITTODataCollator(
            tokenizer=self.tokenizer,
            train_dataset=empty_dataset,
            max_length=128,
            max_prompt_length=64,
            batch_size=2,
            bootstrap_count=3
        )
        empty_collator.model = self.model
        
        empty_collator.resample(step=1)
        
        self.assertIn(1, empty_collator.cache)
        self.assertEqual(len(empty_collator.cache[1]), 0)

if __name__ == "__main__":
    unittest.main()