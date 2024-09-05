# User Profile Recommendation

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
