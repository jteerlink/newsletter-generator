import ollama
import time
import csv

# Define test prompts
prompts = [
    "What is the capital of France?",
    "Summarize the theory of relativity in one paragraph.",
    "Write a haiku about the ocean.",
    "List three benefits of regular exercise.",
    "Explain quantum computing in simple terms.",
]

models = ['llama3', 'gemma3n', 'deepseek-r1']

results = []

for model in models:
    print(f"\nBenchmarking model: {model}\n")
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")
        start = time.time()
        error = ""
        try:
            response = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            output = response["message"]["content"]
        except Exception as e:
            output = ""
            error = str(e)
        end = time.time()
        duration = end - start
        output_length = len(output)
        snippet = output[:80].replace("\n", " ") if output else ""
        anomaly = ""
        # Flag anomalies based on output length
        # Responses shorter than 20 characters are considered 'very short' (may indicate a problem)
        # Responses longer than 1000 characters are considered 'very long' (may be verbose or off-topic)
        if not output:
            anomaly = "empty response"
        elif output_length < 20:
            anomaly = "very short"
        elif output_length > 1000:
            anomaly = "very long"
        print(f"Response: {snippet}{'...' if output_length > 80 else ''}")
        print(f"Time taken: {duration:.2f} seconds | Output length: {output_length} | Error: {error} | Anomaly: {anomaly}\n")
        results.append({
            "model": model,
            "prompt_id": i+1,
            "prompt_text": prompt,
            "response_time": f"{duration:.2f}",
            "output_length": output_length,
            "error": error,
            "anomaly": anomaly,
            "response_snippet": snippet,
        })

# Save results to CSV
with open("benchmark_results.csv", "w", newline="") as csvfile:
    fieldnames = ["model", "prompt_id", "prompt_text", "response_time", "output_length", "error", "anomaly", "response_snippet"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nBenchmarking complete. Results saved to benchmark_results.csv.")
