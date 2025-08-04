# Synergistic Self-Correction for Mathematical Reasoning

This repository contains the code and models for the research project "Synergistic Self-Correction for Mathematical Reasoning," conducted at the Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT).

**Faculty Mentor:** Dr. Abhishek Jindal  
**Author:** Pratham Patel

---

## Abstract

Large Language Models (LLMs) often struggle with complex, multi-step reasoning tasks that require high degrees of accuracy. This project introduces Synergistic Self-Correction (S2C), a multi-stage, structured inference framework designed to enhance an LLM's reasoning capabilities by simulating an internal cognitive ensemble. The pipeline decomposes problem-solving into three distinct functional stages: **Generation**, **Adversarial Critique**, and **Verified Synthesis**. Our evaluation on the GSM8K benchmark, using a fine-tuned Llama-3-8B-Instruct model, demonstrates a significant **60% relative improvement** in problem-solving accuracy, validating the efficacy of the S2C framework.

---

## The S2C Framework

The core of this research is the Synergistic Self-Correction (S2C) pipeline. Instead of generating an answer in a single pass, the model is trained to adopt three distinct "personas" to systematically deconstruct, analyze, and refine its own solutions.

1.  **Generate:** The model first provides a step-by-step solution to the problem, breaking down its logic into "Critical Points."
2.  **Critique:** It then re-examines its own solution, actively searching for potential flaws, errors, or unstated assumptions in the Critical Points.
3.  **Synthesize:** Finally, the model integrates the feedback from the critique stage to produce a final, verified answer.

This structured, internal monologue allows the model to catch and correct its own errors, leading to more robust and reliable reasoning.

---

## Key Results

The S2C framework was applied to a `meta-llama/Meta-Llama-3-8B-Instruct` model and trained using a hybrid strategy of Supervised Fine-Tuning and Proximal Policy Optimization (PPO) on the **GSM8K** dataset.

The final S2C-enhanced model achieved a **60% relative improvement in accuracy** on the GSM8K test set compared to the base model.

![Benchmark Results](./s2c_benchmark_results.png)

The learning progress during the PPO phase is captured by the mean reward, which directly corresponds to the rate of correctly solved problems.

![Mean PPO Reward](./graphs/ppo_returns_mean.png)

---

## Repository Structure

```
.
├── s2c_llama3_8b_final/      # Final PEFT adapter for the S2C model.
├── s2c_llama3_8b_checkpoints/  # Intermediate training checkpoints.
├── graphs/                     # Graphs generated from TensorBoard logs.
├── logs/                       # TensorBoard training logs.
├── train_s2c_rl.py             # Script to resume PPO training on the model.
├── evaluate_s2c.py             # Script to benchmark the models and generate the results graph.
├── benchmark_models.py         # Script to test the base model and environment setup.
├── final_report.pdf            # The full research paper.
└── README.md                   # This file.
```

---

## How to Reproduce

### 1. Setup

First, clone the repository and install the required Python libraries.

```bash
git clone <repository-url>
cd <repository-name>
pip install torch transformers datasets peft trl bitsandbytes matplotlib sentencepiece
```

### 2. Hugging Face Authentication

You will need a Hugging Face access token to download the Llama-3 model.

```bash
huggingface-cli login
# Or set the environment variable
export HF_TOKEN="your_hf_token_here"
```
*Note: The scripts in this repository might contain a hardcoded token for convenience, but using the methods above is best practice.*

### 3. Evaluate Models

To replicate the benchmark results, run the evaluation script. This will test the base Llama-3 model, the step 200 checkpoint, and the final S2C model on a sample of the GSM8K test set.

The script will print the scores to the console and save the bar chart as `s2c_benchmark_results.png`.

```bash
python evaluate_s2c.py
```

### 4. Resume Training

The `train_s2c_rl.py` script is configured to resume training from the `step_200` checkpoint.

```bash
python train_s2c_rl.py
```

This will:
- Load the base Llama-3 model and apply the adapter from `s2c_llama3_8b_checkpoints/step_200/`.
- Continue the PPO training loop.
- Save new checkpoints to the `s2c_llama3_8b_checkpoints/` directory.
- Save the final trained adapter to `s2c_llama3_8b_final/`.
- Log training metrics to the `logs/` directory for TensorBoard.

---

## Citation

This work was conducted as part of the Summer Research Internship (SRI) program at DA-IICT.

If you use this work, please cite the author and the institution:
- **Author:** Pratham Patel
- **Institution:** Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT)
- **Mentor:** Dr. Abhishek Jindal
