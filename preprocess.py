import pickle
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from collections import defaultdict
import statistics
from math import sqrt
import heapq
import json
import random
from sklearn.model_selection import train_test_split
import argparse
import os
import torch
import numpy as np
import math


parser = argparse.ArgumentParser(description="Find top k values in features")
parser.add_argument("--k", type=int, default=5, help="How many features to extract.")
parser.add_argument(
    "--input_directory",
    type=str,
    default="datasets/Amazon/MoviesAndTV",
    help="The dataset to use.",
)
parser.add_argument(
    "--pickled_reviews", type=str, default="reviews.pickle", help="pickled review data."
)
parser.add_argument("--items", type=str, default="item.json", help="item metadata.")
parser.add_argument(
    "--output",
    type=str,
    default="amazon_k_best_features.json",
    help="output file.",
)

args = parser.parse_args()

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def normalize_rating(rating):
    """
    Normalize scale 1 to 5 -> -1 to 1
    Input: Score 1 to 5
    Output: Score -1 to 1
    """
    return 2 * (rating - 1) / (5 - 1) - 1


class User:
    def __init__(self, user_id, data):
        self.user_id = user_id
        self.data = data
        (
            self.positive_reviews,
            self.negative_reviews,
            self.ratings,
            self.stems,
            self.all_ratings,
        ) = self.retrieve_features(self.data)

    def retrieve_features(self, data):
        positive_reviews = defaultdict(list)
        negative_reviews = defaultdict(list)
        ratings = defaultdict(list)
        stems = defaultdict(set)
        all_ratings = []
        for review in data:
            feature = review["feature"]
            stem_feature = stemmer.stem(feature)
            if (
                stem_feature == "movi"
                or stem_feature == "film"
                or len(stem_feature) < 2
            ):
                continue
            stems[stem_feature].add(feature)
            normalized_rating = normalize_rating(review["label"])
            ratings[stem_feature].append(normalized_rating)
            if normalized_rating < 0:
                negative_reviews[stem_feature].append(review["review"])
            else:
                positive_reviews[stem_feature].append(review["review"])
            all_ratings.append(normalized_rating)
        return positive_reviews, negative_reviews, ratings, stems, all_ratings

    def utility_score(self, feature):
        feature_rating = self.ratings[feature]
        inferred_rating = sum(feature_rating) / len(feature_rating)
        coverage = len(feature_rating) / len(self.data)
        var_all_items = 0 if len(self.all_ratings) == 1 else statistics.variance(self.all_ratings)
        significance = min(
            2, abs(inferred_rating) * (var_all_items / (sqrt(len(feature_rating))))
        )
        return coverage * significance * inferred_rating

    def k_best_features(self, k=15):
        features = {}
        for stem in self.stems:
            features[stem] = self.utility_score(stem)
        k_best_features = heapq.nlargest(k, features.items(), key=lambda i: abs(i[1]))

        k_best_data = []
        for stem, score in k_best_features:
            k_best_data.append(
                {
                    "stem": stem,
                    "features": list(self.stems[stem]),
                    "score": score,
                    "reviews": self.negative_reviews[stem]
                    if score < 0
                    else self.positive_reviews[stem],
                }
            )
        return {"user_id": self.user_id, "k_best_data": k_best_data}


if __name__ == "__main__":
    with open(
        os.path.join(args.input_directory, args.pickled_reviews), "rb"
    ) as file_handle:
        reviews = pickle.load(file_handle)

    with open(os.path.join(args.input_directory, args.items), "r") as f:
        items = json.load(f)

    items_with_titles = {}

    if args.input_directory == "datasets/TripAdvisor":
        for item in items:
            item_id = item["hotelID"]
            items_with_titles[item_id] = f"{item['hotelTitle']}; {item['hotelCity']}"

    else:
        for item in items:
            if "title" in item:
                title = item["title"]
                item_id = item["item"]
                items_with_titles[item_id] = title
    
    stemmer = PorterStemmer()
    user_reviews = defaultdict(list)

    # remove items without a title
    for idx, review in enumerate(reviews):
        # if review["item"] in items_with_titles:
        review["index"] = idx
        user_reviews[review["user"]].append(review)

    # print(f"pre-filtered number of users: {len(user_reviews)}")
    # user_reviews = {
    #     k: v for k, v in user_reviews.items() if len(v) >= 10
    # }  # remove users who have less than 20 reviews

    num_users = set()
    num_items = set()
    num_explanations = set()
    data = []
    for _, rev in user_reviews.items():
        for review in rev:
            if 'template' not in review:
                if 'sentence' in review:
                    review['template'] = review['sentence'][0]
                else:
                    continue
            # review["title"] = items_with_titles[review["item"]]
            num_users.add(review["user"])
            num_items.add(review["item"])
            num_explanations.add(review["template"][2])
            data.append(
                {
                    "user": review["user"],
                    "item": review["item"],
                    "label": review["rating"],
                    "review": review["template"][2],
                    "feature": review["template"][0],
                    # "title": review["title"],
                    "index": review["index"],
                }
            )

    print(f"Number of Users: {len(num_users)}")
    print(f"Number of Items: {len(num_items)}")
    print(f"Number of Explanations: {len(num_explanations)}")
    print(f"Number of Records {len(data)}")

    # Group data by user
    user_data = defaultdict(list)
    for item in data:
        user_data[item['user']].append(item)

    # Filter users with fewer than 6 occurrences
    filtered_user_data = {user: items for user, items in user_data.items() if len(items) >= 10}
    X = []
    for user, items in filtered_user_data.items():
        for item in items:
            X.append(item)
    print('filtered number of users:', len(filtered_user_data))
    print('filtered number of items:', len(X))

    X_train, X_test, X_val = [], [], []
    total_items = sum([len(items) for items in filtered_user_data.values()])
    train_target = 0.8 * total_items
    test_val_target = total_items - train_target

    train_count, test_val_count = 0, 0

    for user, items in filtered_user_data.items():
        random.shuffle(items)  # Shuffle the user's items
        # Reserve 5 items for training and distribute the rest
        train_items = items[:5]
        remaining_items = items[5:]

        # Calculate remaining distribution targets
        remaining_train_target = max(0, train_target - train_count - len(train_items))
        remaining_test_val_target = max(0, test_val_target - test_val_count)

        # Distribute the remaining items
        if remaining_train_target > 0:
            additional_train_items = min(len(remaining_items), int(np.round(remaining_train_target / remaining_test_val_target * len(remaining_items))))
            train_items.extend(remaining_items[:additional_train_items])
            remaining_items = remaining_items[additional_train_items:]

        # Split the rest between test and validation
        if len(remaining_items) == 1:
            # Randomly assign if only one item left
            if random.random() < 0.5:
                X_val.extend(remaining_items)
            else:
                X_test.extend(remaining_items)
        elif len(remaining_items) > 1:
            mid_point = len(remaining_items) // 2
            X_val.extend(remaining_items[:mid_point])
            X_test.extend(remaining_items[mid_point:])

        X_train.extend(train_items)
        train_count += len(train_items)
        test_val_count += len(remaining_items)

    # Assertions
    train_users = {}
    train_items = {}
    for i in X_train:
        user_id = i['user']
        item_id = i['item']
        train_users[user_id] = train_users.get(user_id, []) + [i]
        train_items[item_id] = train_items.get(item_id, 0) + 1
    assert all(len([x for x in X_train if x['user'] == user]) >= 5 for user in train_users)
    assert all(user in train_users for user in (x['user'] for x in X_val))
    assert all(user in train_users for user in (x['user'] for x in X_test))

    print("Train Set:", len(X_train))
    print("Test Set:", len(X_test))
    print("Validation Set:", len(X_val))

    train_index = []
    validation_index = []
    test_index = []
    # Open a file for writing
    with open(os.path.join(args.input_directory, "train.jsonl"), "w") as f:
        for entry in X_train:
            # Convert the dictionary to a JSON-formatted string
            json_str = json.dumps(entry)
            # Write the JSON string to the file, followed by a newline
            f.write(json_str + "\n")
            train_index.append(str(entry["index"]))
    with open(os.path.join(args.input_directory, "validation.jsonl"), "w") as f:
        for entry in X_val:
            # Convert the dictionary to a JSON-formatted string
            json_str = json.dumps(entry)
            # Write the JSON string to the file, followed by a newline
            f.write(json_str + "\n")
            validation_index.append(str(entry["index"]))
    with open(os.path.join(args.input_directory, "test.jsonl"), "w") as f:
        for entry in X_test:
            # Convert the dictionary to a JSON-formatted string
            json_str = json.dumps(entry)
            # Write the JSON string to the file, followed by a newline
            f.write(json_str + "\n")
            test_index.append(str(entry["index"]))

    for split in ["train", "validation", "test"]:
        with open(os.path.join(args.input_directory, f"{split}.index"), "w") as f:
            if split == "train":
                f.write(" ".join(train_index))
            # elif split == "validation":
            #     f.write(" ".join(validation_index))
            else:
                f.write(" ".join(test_index))

    data = []
    for user_id, user_review in tqdm(train_users.items(), total=len(train_users)):
        user = User(user_id, user_review)
        k_best_features = user.k_best_features(k=args.k)
        data.append(k_best_features)

    with open(f"user_profiles/{args.output}", "w") as f:
        json.dump(data, f, indent=2)
