import os
import sys
import json
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import hydra
from google import genai
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "evaluations"

def retrieve_results(job_name: str) -> list[dict[str, Any]]:
    client = genai.Client()
    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name

            file_content_bytes = client.files.download(file=result_file_name)
            file_content_str = file_content_bytes.decode('utf-8')
            
            json_objects = []
            
            for line in file_content_str.splitlines():
                json_objects.append(json.loads(line))
            
            with open(OUTPUT_PATH / "batch_results.json", 'w', encoding='utf-8') as f:
                json.dump(json_objects, f, indent=4) 
            
            return json_objects
                    
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
        sys.exit(0)

def process_leaderboard(matches: list[dict[str, Any]]) -> dict[str, dict[str, list[int, int]]]:
    categories = {}

    for match in matches:
        key = match["key"]
        data_authid, matchup = key.split("__")
        if data_authid not in categories.keys():
            categories[data_authid] = {}

        model_a, model_b = matchup.split("_VS_")

        if model_a not in categories[data_authid].keys():
            categories[data_authid][model_a] = [0, 0]

        if model_b not in categories[data_authid].keys():
            categories[data_authid][model_b] = [0, 0]

        winner = json.loads(match["response"]["candidates"][0]["content"]["parts"][0]["text"])["answer"]

        categories[data_authid][model_a][winner != "A"] += 1
        categories[data_authid][model_b][winner == "A"] += 1

    with open(OUTPUT_PATH / "match_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=4)

    return categories

def calculate_winrate(statistics: dict[str, dict[str, list[int, int]]]) -> dict[str, dict[str, float]]:
    percentage = statistics.copy()
    for category, stats in statistics.items():
        for model, stat in stats.items():
            percentage[category][model] = stat[0] / (stat[0] + stat[1])
    
    with open(OUTPUT_PATH / "winrate.json", 'w', encoding='utf-8') as f:
        json.dump(percentage, f, indent=4)


@hydra.main(version_base=None, config_path="../configs", config_name="conclude")
def main(config: DictConfig):
    load_dotenv()
    os.environ["api_key"] = os.environ.get("GEMINI_API_KEY")

    assert config.job_name, "Job name can't be null; insert arg job_name=<job_name> when running the script."

    matches = retrieve_results(config.job_name)
    statistics = process_leaderboard(matches)
    calculate_winrate(statistics)


if __name__ == "__main__":
    main()