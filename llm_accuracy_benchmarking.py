import json
import time
import os
import string
import csv
import argparse
import subprocess
from jtop import jtop  # Jetson stats

def normalize(text):
    return text.strip().lower().translate(str.maketrans("", "", string.punctuation))

def run_ollama_cli(model, prompt, image_path):
    """Run Ollama CLI for one multimodal query and return output + latency."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt, image_path],
            capture_output=True,
            text=True,
            check=True
        )
        latency = time.time() - start
        output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        # enforce one word/number
        enforced = output.strip().split()[0].lower() if output else ""
        return enforced, latency
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama: {e.stderr}", flush=True)
        return "", 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--image-dir", default="Images_LR")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--index", type=int, required=True, help="Question index to process")
    args = parser.parse_args()

    # auto-name output if not provided
    if args.output is None:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        args.output = f"benchmark_results_{safe_model}.csv"

    # load dataset
    with open(args.questions) as f:
        all_questions = json.load(f)["questions"]
    with open(args.answers) as f:
        all_answers = {a["id"]: a for a in json.load(f)["answers"]}

    if args.index >= len(all_questions):
        print(f"❌ Index {args.index} out of range", flush=True)
        return

    q = all_questions[args.index]
    qid = q["id"]
    img_id = q["img_id"]
    image_path = os.path.join(args.image_dir, f"{img_id}.tif")

    if not os.path.exists(image_path):
        print(f"⚠️ Missing image {image_path}, skipping.", flush=True)
        return

    question_text = q["question"] + "\nAnswer with exactly one word or number only. Do not explain."
    gt_answers = [
        normalize(all_answers[aid]["answer"])
        for aid in q.get("answers_ids", [])
        if aid in all_answers
    ]
    if not gt_answers:
        print(f"⚠️ No ground truth for qid {qid}, skipping.", flush=True)
        return

    # ---- Run inference + track power ----
    power_samples = []
    with jtop() as jetson:
        if not jetson.ok():
            print("⚠️ jtop not ready", flush=True)
            return

        # start inference
        response, latency = run_ollama_cli(args.model, question_text, image_path)

        # sample power during inference
        end_time = time.time() + latency
        while time.time() < end_time and jetson.ok():
            p = jetson.power
            if p:
                tot_power = p["tot"].get("power", 0) / 1000.0  # W
                tot_avg   = p["tot"].get("avg", 0) / 1000.0
                rail_cpu_gpu = p["rail"]["VDD_CPU_GPU_CV"].get("power", 0) / 1000.0
                rail_soc     = p["rail"]["VDD_SOC"].get("power", 0) / 1000.0

                power_samples.append({
                    "tot": tot_power,
                    "tot_avg": tot_avg,
                    "cpu_gpu": rail_cpu_gpu,
                    "soc": rail_soc
                })
            time.sleep(0.2)

        # grab a few extra samples after inference
        for _ in range(5):
            if jetson.ok():
                p = jetson.power
                if p:
                    tot_power = p["tot"].get("power", 0) / 1000.0
                    tot_avg   = p["tot"].get("avg", 0) / 1000.0
                    rail_cpu_gpu = p["rail"]["VDD_CPU_GPU_CV"].get("power", 0) / 1000.0
                    rail_soc     = p["rail"]["VDD_SOC"].get("power", 0) / 1000.0

                    power_samples.append({
                        "tot": tot_power,
                        "tot_avg": tot_avg,
                        "cpu_gpu": rail_cpu_gpu,
                        "soc": rail_soc
                    })
            time.sleep(0.2)

    # compute power stats
    if power_samples:
        avg_tot = sum(p["tot"] for p in power_samples) / len(power_samples)
        max_tot = max(p["tot"] for p in power_samples)
        avg_cpu_gpu = sum(p["cpu_gpu"] for p in power_samples) / len(power_samples)
        max_cpu_gpu = max(p["cpu_gpu"] for p in power_samples)
    else:
        avg_tot = max_tot = avg_cpu_gpu = max_cpu_gpu = 0.0

    is_correct = normalize(response) in gt_answers

    # ---- Append result to CSV ----
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "question_id", "latency_sec", "correct",
                "model_response", "ground_truth", "question_text",
                "avg_tot_w", "max_tot_w",
                "avg_cpu_gpu_w", "max_cpu_gpu_w"
            ])
        writer.writerow([
            qid, f"{latency:.3f}", int(is_correct),
            response, "|".join(gt_answers), q["question"],
            f"{avg_tot:.2f}", f"{max_tot:.2f}",
            f"{avg_cpu_gpu:.2f}", f"{max_cpu_gpu:.2f}"
        ])
        f.flush()

    # ---- Print results (unbuffered) ----
    print(f"[Q{qid}] {q['question']}", flush=True)
    print(f"Model: {response}", flush=True)
    print(f"GT: {gt_answers}", flush=True)
    print(f"Correct: {is_correct}, Time: {latency:.2f}s", flush=True)
    print(f"Power: tot avg {avg_tot:.2f} W, tot max {max_tot:.2f} W, "
          f"cpu+gpu avg {avg_cpu_gpu:.2f} W, cpu+gpu max {max_cpu_gpu:.2f} W", flush=True)

if __name__ == "__main__":
    main()

