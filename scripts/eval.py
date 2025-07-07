import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an impartial evaluator.
Below is a sample of a human author's writing and two options.

### HUMAN AUTHOR'S WRITING:
{demo}

### OUTPUT A:
{text_a}

### OUTPUT B:
{text_b}

### Task
Which option was written by the human author based on similarity to the HUMAN AUTHOR'S WRITING above? Respond only with a JSON of the following format:
{{
"answer": "<The option most similar to the HUMAN AUTHOR'S WRITING; either A or B>",
"reasoning": "<Brief explanation of why this option is more similar>"
}}

ALWAYS REMAIN IMPARTIAL WHEN EVALUATING OUTPUTS.
"""

def load_json_file(file_path: Path) -> Dict[str, Any]:
    logger.info(f"Loading data from {file_path}")
    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def find_matching_samples(results_a: Dict, results_b: Dict) -> List:
    prompts_a = {sample["prompt"]: sample for sample in results_a.get("outputs",)}
    matches = []
    for sample_b in results_b.get("outputs",):
        prompt = sample_b.get("prompt")
        if prompt in prompts_a:
            matches.append((prompts_a[prompt], sample_b))
    logger.info(f"Found {len(matches)} matching samples.")
    return matches

def create_batch_input_file(
    matching_samples: List,
    demo_text: str,
    config: DictConfig,
    output_path: Path
):
    logger.info(f"Creating batch input file at: {output_path}")
    with output_path.open('w', encoding='utf-8') as f:
        for i, (sample_a, sample_b) in enumerate(matching_samples):
            prompt_text = PROMPT_TEMPLATE.format(
                demo=demo_text,
                text_a=sample_a["generated_response"],
                text_b=sample_b["generated_response"]
            )
            request = {
                "key": f"eval_{i}",
                "request": {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generation_config": {
                        "temperature": config.gemini.temperature,
                        "max_output_tokens": config.gemini.max_output_tokens
                    },
                    "system_instruction": {
                        "parts": [{"text": config.gemini.system_instruction}]
                    }
                }
            }
            f.write(json.dumps(request) + '\n')
    logger.info(f"Successfully created batch input file with {len(matching_samples)} requests.")

def _parse_response_text(response_text: str) -> Dict[str, str]:
    try:
        if response_text.strip().startswith('```json'):
            response_text = response_text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(response_text)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse JSON response: {e}\nRaw response: '{response_text}'")
        return {"answer": "PARSE_ERROR", "reasoning": str(e)}

def process_batch_results(
    client: genai.Client,
    completed_job: Any,
    matching_samples: List
) -> List:
    """Downloads and processes the results from a completed batch job."""
    if not (completed_job.dest and completed_job.dest.file_name):
        raise RuntimeError("Job completed but no output file was found.")

    result_file_name = completed_job.dest.file_name
    logger.info(f"Downloading results from file: {result_file_name}")
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')

    samples_by_key = {f"eval_{i}": (sa, sb) for i, (sa, sb) in enumerate(matching_samples)}
    evaluations = []

    for line in file_content.splitlines():
        if not line:
            continue
        
        result_data = json.loads(line)
        request_key = result_data.get("key")
        sample_a, sample_b = samples_by_key.get(request_key, ({}, {}))

        evaluation = {
            "key": request_key,
            "prompt": sample_a.get("prompt"),
            "text_a": sample_a.get("generated_response"),
            "text_b": sample_b.get("generated_response"),
        }

        if "response" in result_data:
            response_text = result_data["response"]["candidates"]["content"]["parts"]["text"]
            parsed_result = _parse_response_text(response_text)
            evaluation["evaluation"] = {
                "answer": parsed_result.get("answer", "MISSING_ANSWER"),
                "reasoning": parsed_result.get("reasoning", "MISSING_REASONING"),
                "raw_response": response_text
            }
        elif "status" in result_data:
            error_message = result_data["status"]["message"]
            logger.error(f"Error for request '{request_key}': {error_message}")
            evaluation["evaluation"] = {"answer": "API_ERROR", "reasoning": error_message}
        
        evaluations.append(evaluation)
        
    return evaluations

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(config: DictConfig):
    """Main function to run the evaluation pipeline."""
    load_dotenv()
    OmegaConf.resolve(config)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    results_a_path = Path(hydra.utils.get_original_cwd()) / config.evaluation.results_a_path
    results_b_path = Path(hydra.utils.get_original_cwd()) / config.evaluation.results_b_path

    results_a = load_json_file(results_a_path)
    results_b = load_json_file(results_b_path)

    matching_samples = find_matching_samples(results_a, results_b)
    if not matching_samples:
        raise ValueError("No matching samples found. Aborting.")

    if config.evaluation.max_samples > 0:
        matching_samples = matching_samples[:config.evaluation.max_samples]
        logger.info(f"Limiting to {len(matching_samples)} samples for evaluation.")

    batch_input_file = output_dir / "batch_requests.jsonl"
    create_batch_input_file(matching_samples, config.evaluation.demo_text, config, batch_input_file)

    logger.info("Uploading batch input file...")
    uploaded_file = client.files.upload(file_path=str(batch_input_file))
    
    job_display_name = config.batch.job_name or f"evaluation-job-{int(time.time())}"
    logger.info(f"Submitting batch job '{job_display_name}'...")
    batch_job = client.batches.create(
        model=config.gemini.model,
        src=uploaded_file.name,
        config={'display_name': job_display_name},
    )
    logger.info(f"Batch job created: {batch_job.name}")

    logger.info("Waiting for batch job to complete...")
    while batch_job.state.name not in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED'):
        time.sleep(config.batch.check_interval)
        batch_job = client.batches.get(name=batch_job.name)
        logger.info(f"Current job state: {batch_job.state.name}")

    if batch_job.state.name == 'JOB_STATE_FAILED':
        raise RuntimeError(f"Batch job failed. Error: {batch_job.error}")

    logger.info("Batch job completed successfully!")

    evaluations = process_batch_results(client, batch_job, matching_samples)
    
    a_wins = sum(1 for e in evaluations if e.get("evaluation", {}).get("answer") == "A")
    b_wins = sum(1 for e in evaluations if e.get("evaluation", {}).get("answer") == "B")
    errors = len(evaluations) - a_wins - b_wins

    summary = {
        "config": OmegaConf.to_container(config, resolve=True),
        "batch_job_name": batch_job.name,
        "total_evaluations": len(evaluations),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "errors": errors,
        "a_win_rate": a_wins / len(evaluations) if evaluations else 0,
        "b_win_rate": b_wins / len(evaluations) if evaluations else 0,
        "evaluations": evaluations
    }

    output_path = output_dir / (config.evaluation.output_filename or "evaluation_results.json")
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_path}")

    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"A wins: {a_wins} ({summary['a_win_rate']:.1%})")
    logger.info(f"B wins: {b_wins} ({summary['b_win_rate']:.1%})")
    logger.info(f"Errors: {errors} ({(errors / len(evaluations) if evaluations else 0):.1%})")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()