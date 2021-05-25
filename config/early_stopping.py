import pickle


early_stopping_labels = ["monitor", "min_delta", "patience", "verbose", "restore_best_weights"]

with open('config/early_stopping.pickle', 'rb') as handle:
    early_stopping = pickle.load(handle)


def save():
    with open('config/early_stopping.pickle', 'wb') as handle:
        pickle.dump(early_stopping, handle)

