# User Profile Recommendation

## UPDATE: Reproducing Transparent and Scrutable Recommendations 
[![Paper arXiv](https://img.shields.io/badge/arXiv-2609.19831-B31B1B.svg)](https://arxiv.org/abs/2609.19831)
[![BlackboxNLP @ EMNLP](https://img.shields.io/badge/BlackboxNLP-Accepted-blue.svg)](https://blackboxnlp.github.io/)

We welcome you to explore the repository that contains the [official codebase](https://github.com/nmamie/transparent_user_profiles) for the follow-up study [*Reproducing Transparent and Scrutable Recommendations: Exploring Open-Weight Models via Natural-Language User Profiles*](https://arxiv.org/abs/2609.19831). It provides a comprehensive reproduction, codebase fixes, and empirical extensions of the User Profile Recommendation (UPR) framework.

### 🚀 Environment Setup & Quick Start

Follow these steps to create the environment, install dependencies, and run the pipeline:

#### Option A: Using `uv` (Recommended)

```bash
# 1. Initialize project structure
mkdir -p src/user_profiles_neurosymbolic_ai
touch src/user_profiles_neurosymbolic_ai/__init__.py

# 2. Create virtual environment and install dependencies with uv
uv venv .venv
uv sync

# 3. Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Make shell execution scripts executable
chmod +x train_amazon_comb.sh train_tripadvisor_comb.sh train_amazon_seeds.sh

# 5. Run batch experiment scripts
./train_amazon_comb.sh
./train_tripadvisor_comb.sh
./train_amazon_seeds.sh
```

#### Option B: Using Standard Python `venv` & `pip`

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Make scripts executable and run
chmod +x train_amazon_comb.sh train_tripadvisor_comb.sh train_amazon_seeds.sh
./train_amazon_comb.sh
```

## Datasets

For Movies And TV, use the following datasets:

```
datasets/Amazon/MoviesAndTV/train.jsonl
datasets/Amazon/MoviesAndTV/validation.jsonl
datasets/Amazon/MoviesAndTV/test.jsonl
```

For TripAdvisor, use the following datasets:

```
datasets/TripAdvisor/train.jsonl
datasets/TripAdvisor/validation.jsonl
datasets/TripAdvisor/test.jsonl
```

For more information, please see (https://github.com/lileipisces/PETER) for the entire original data source and information on how to use the Sentires toolkit.

## User Profiles

All user profiles can be found in the `user_profiles` directory. The number indicates how many features were used to generate that profile. Profiles without a number were generated using 5 features. In addition, we provide both Llama2-7B and Mistral-7B profiles.

The main profiles used for testing is

```
amazon_profiles.json
amazon_profiles_mistral.json
trip_advisor_profiles.json
trip_advisor_profiles_mistral.json
```

## Recommendation Baselines

To run the available baselines on Cornac,

> `rec_baselines.py`

```
python rec_baselines.py -d Amazon/MoviesAndTV
```

## Generating User Profiles

To rerun our experiments, you can use the saved user profiles in the `user_profiles` directory. To generate the user profiles from scratch, follow the instructions below.

### Preprocess

To preprocess the data, run:

```
python preprocess.py
```

### Generate NL Profiles

To generate the profiles, run the following command with the appropriate LLM.

```
python generate_profile.py
```

## Training

Here is an example to fine-tune the model:

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--output_dir out/amazon-out \
--lr 0.0003 \
--batch_size 8 \
--num_train_epochs 5 \
--seed 42
```

## Scrutability Test

To generate scrutable profiles use

```
python generate_counterfactual_profiles.py
```

The few shot prompts used are contained in the file above.

## Evaluation

```
python evaluate.py
```

can by use to evaluate the recommender performance of the test set or sampling files. Please be sure to use the correct user profiles and pretrained model during evaluation.

# Code writing assistants

To cite our work, please cite:

```
@inproceedings{ramos-etal-2024-transparent,
    title = "Transparent and Scrutable Recommendations Using Natural Language User Profiles",
    author = "Ramos, Jerome  and
      Rahmani, Hossein A.  and
      Wang, Xi  and
      Fu, Xiao  and
      Lipani, Aldo",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.753",
    pages = "13971--13984",
}
```

To cite the reproduction study, please cite:

```bibtex
@inproceedings{mamié2026reproducingtransparentscrutablerecommendations,
      title={Reproducing Transparent and Scrutable Recommendations: Exploring Open-Weight Models via Natural-Language User Profiles}, 
      author={\textbf{Noah Mamié} and Laurin van den Bergh},
      note={Accepted at BlackBoxNLP@EMNLP'26 (The 9th BlackboxNLP Workshop Special Track: Reproducibility and Reliability in Interpretability Analyses)},
      year={2026},
      eprint={2609.19831},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2609.19831}, 
}
```
