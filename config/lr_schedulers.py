import pickle


lr_schedulers_labels = ["monitor", "min_delta", "factor", "patience", "min_lr", "verbose"]


with open('config/lr_schedulers.pickle', 'rb') as handle:
    lr_schedulers = pickle.load(handle)

def save():
    with open('config/lr_schedulers.pickle', 'wb') as handle:
        pickle.dump(lr_schedulers, handle)
