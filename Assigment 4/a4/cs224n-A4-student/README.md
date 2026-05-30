# CS224n Assignment 4 - LLM Evals

In this assignment, you will evaluate the properties of various different LLMs using
standard benchmarking techniques, implement an LLM-as-a-judge evaluation, and explore
red-teaming approaches.

## Install

First, create a new conda environment with Python 3.10:

```bash
conda create -n cs224n-A4 python=3.10
```

Activate the environment:

```bash
conda activate cs224n-A4
```

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Set up your environment:**

This assignment uses **Google Vertex AI (Gemini)** via your local Google Cloud credentials (e.g., `gcloud auth application-default login`).

Create a local `.env` file (or export env vars in your shell) with:

- `GCP_PROJECT_NAME`: your GCP project ID (e.g., `hellow-world-485923-x5`) (used to initialize the Vertex AI client)
- `STUDENT_EMAIL`: only needed for models **G/H/I** (used to deterministically seed the per-student password)

For an example see `example_usage.py`.

**Run the example:**

```bash
python example_usage.py
```
