import json

INPUT_PATH = "./results/per_pathology_results.json"
OUTPUT_PATH = "./results/per_pathology_results_clean.json"

# Keys to remove (heavy + unreadable)
REMOVE_KEYS = ["gt_scores", "top1_scores", "score_gaps"]


def clean_metrics(metrics):
    """Remove unwanted keys from metrics dict"""
    if metrics is None:
        return None

    for key in REMOVE_KEYS:
        if key in metrics:
            del metrics[key]
    return metrics


def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    cleaned_data = {}

    for pathology, content in data.items():
        scenarios = content.get("scenarios", {})

        cleaned_scenarios = {}

        for scenario_name, scenario_data in scenarios.items():

            # ---- HARD CASES ----
            if scenario_name == "hard_cases":
                metrics = scenario_data.get("metrics", None)

                cleaned_scenarios[scenario_name] = {
                    "num_samples": scenario_data.get("num_samples", 0),
                    "metrics": clean_metrics(metrics)
                }

            # ---- FULL / PARTIAL ----
            else:
                cleaned_scenarios[scenario_name] = clean_metrics(scenario_data)

        cleaned_data[pathology] = {
            "num_samples": content.get("num_samples", 0),
            "scenarios": cleaned_scenarios
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Cleaned JSON saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()