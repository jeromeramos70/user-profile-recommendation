Codes for creating the datasets: https://github.com/lileipisces/Sentires-Guide
Codes for NETE in CIKM'20, based on GRU: https://github.com/lileipisces/NETE
Codes for PETER in ACL'21, based on small unpretrained Transformer: https://github.com/lileipisces/PETER
Codes for PEPLER in 2022, based on large pretrained GPT-2: https://github.com/lileipisces/PEPLER
Codes for an ecosystem for Recommender Systems-based Natural Language Generation: https://github.com/lileipisces/NLG4RS

reviews.pickle can be loaded via pickle package as a python list, where each record is a python dict in the form of
{'user': '7B555091EC0818119062CF726B9EF5FF',  # str
'item': '1068719',  # str
'rating': 5,  # float or int
'template': ('train', 'you', 'the train from the airport lets you off right inside the mall that the hotel is in', 1),  # a tuple in the form of (feature, adjective, sentence, sentiment) where the first three are str, while the last one is int with values of -1 or 1
'predicted': 'vibe'}  # str, predicted feature for this user-item pair via PMI

folders named 1, 2, 3, 4 and 5 are data splits
each folder contains train, validation and test files which indicate the indices of their records in the list of reviews.pickle
xx.index contains a line of numbers (indices), e.g., 5 8 9 10

If you simply want to reproduce the experiments reported in our paper, please ignore xx.json because they are meant for other purposes, e.g., teaching or future work.
