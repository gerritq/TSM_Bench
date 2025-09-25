# TSM-Bench: Detecting LLM-Generated Text in Real-World Wikipedia Editing Practices


This repository contains the official implementation of **TSM-Bench: Detecting LLM-Generated Text in Real-World Wikipedia Editing Practices**.

<!-- Add link: [Paper Title]() -->

---

## TSM-Bench

![Overview of our TSM-Bench](assets/overview.png)

Automatically detecting machine-generated text (MGT) is critical to maintain-
ing the knowledge integrity of user-generated content (UGC) platforms such as
Wikipedia. Existing detection benchmarks primarily focus on generic text gen-
eration tasks (e.g., “Write an article about machine learning.”). However, editors
frequently employ LLMs for specific writing tasks (e.g., summarisation). These
task-specific MGT instances tend to resemble human-written text more closely
due to their constrained task formulation and contextual conditioning. In this
work, we show that a range of MGT detectors struggle to identify task-specific
MGT reflecting real-world editing on Wikipedia. We introduce TSM-BENCH, a
multilingual, multi-generator, and multi-task benchmark for evaluating MGT de-
tectors on common, real-world Wikipedia editing tasks. Our findings demonstrate
that (i) average detection accuracy drops by 10–40% compared to prior bench-
marks, and (ii) a generalisation asymmetry exists: fine-tuning on task-specific
data enables generalisation to generic data—even across domains—but not vice
versa. We demonstrate that models fine-tuned exclusively on generic MGT overfit
to superficial artefacts of machine generation. Our results suggest that, in contrast
to prior benchmarks, most detectors remain unreliable for automated detection in
real-world contexts such as UGC platforms. TSM-BENCH therefore provides a
crucial foundation for developing and evaluating future models.

---

## Requirements

### 1. Clone the Repository

```bash
git clone tbd
cd TSM-Bench
```

### 2. Download Pretrained Models

We host the pre-trained models for Experiment 4 (generalisation) on Google Drive. You can also run the models yourself using the script `generalise/code/train_hp_g.sh`.  
The following script will download and unzip the models into `generalise/code/hp_len`.  
The download is ~18GB, unzipped size is ~25GB.

```bash
bash download_models.sh
```

**Note:** This script requires `gdown`. Install it via:

```bash
pip install gdown
```

---

### 3. Set Up the Python Environment

We recommend using a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export HF_HOME="" # optimal to manage hf cache
```

---

## Main Results

**Note:**  
- Experiments 1-2 were run on either a single **NVIDIA A100 80GB** or two **NVIDIA A100 40GB** GPUs. ÷
- Experiments 3-5 were run on a single **NVIDIA A100 40GB**.  
- We strongly recommend using GPUs to replicate results.

### Experiment 1: Off-the shelf detectos

![Figure 3: SOTA Off-the-Shelf Detector Performance](assets/ots.png)

Run the following script to reproduce the results:

```bash
bash run_ots.sh
```

### Experiment 2: Supervised and zero-shot detectors

![Table 1: Within-task Detection (ACC = accuracy, F1 = F1-score)](assets/table2.png)

To run black-box detectors, provide your OpenAI API key. If you skip this, only local models will be evaluated.

**Note:**  
Zero-shot evaluations may take up to 1.5 days. We recommend splitting scripts across HPC jobs.  
Supervised detectors run much faster. To run only those:

```bash
bash detect_train_hp.sh
```

To run all:

```bash
export OPENAI_API_KEY=sk-...
bash run_detection.sh
```

---

### Experiment 3: Feature analysis

![SHAP Values for mDeBERTa trained on task-specific vs generic data](assets/shap_max.png)

To generate the SHAP plot run:

```bash
bash run_shap_vals.sh
```

### Experiment 4: Out-of-domain generalisation

![OOD with mDeberta and GPT4o across domains.](assets/cm_ood_gpt4o.png)

This will populate `generalise/data/detect` with files named:
`trainFile_2_testFile_model_language.jsonl`

```bash
bash run_generalisation.sh
```

---

### Experiment 5: Cross-task generalisation

![Cross-task generalisation with mDeberta and GPT4o across domains.](assets/cm_ct_gpt4o.png)

This will populate `generalise/data/detect` with files named:
`trainFile_2_testFile_model_language.jsonl`

```bash
bash tbd shortly!
```

---

## Other Results

### Linguistic Analysis

![comparison of linguistic features .](assets/la_en.png)

Run the files in `linguistic_analysis/la.sh` with a GPU.

### Prompt Selection

![Prompt Evaluation (values in parentheses show pp improvement over baseline prompts)](assets/table1.png)

You can run this without `QAFactEval` if it causes issues.

To replicate our prompt selection evaluation:

#### 1. Create a Conda Environment for QAFactEval

We recommend using Conda: Clone and install QAFactEval into the current directory:

```bash
conda env create -f environment_qafe.yml
pip install -r requirements_qafe.txt
```

Clone and install QAFactEval into the current directory. Follow setup instructions at: https://github.com/salesforce/QAFactEval. Don't forget to add the `model_folder` in `scorers/qafe.py`.

#### 2. Download Style Classifiers

Same procedure as above. Ensure `gdown` is installed.

```bash
bash download_sc.sh
```

#### 3. Run the Evaluation

This example runs the evaluation for Vietnamese. Adjust the language as needed.

```bash
bash run_prompt_eval.sh
```

---

## Contributing

Valuable contributions include:

- Implementing robust data cleaning with [`mwparserfromhtml`](https://pypi.org/project/mwparserfromhtml/)
- Extending to more languages, adding generators, and expanding task coverage

---

## Citation

> tbd
