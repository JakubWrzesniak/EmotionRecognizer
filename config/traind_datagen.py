import pickle


train_datagen_labels = ["rotation_range", "width_shift_range", "height_shift_range", "shear_range", "zoom_range", "horizontal_flip"]


def save():
    with open('config/trained_datagen.pickle', 'wb') as handle:
        pickle.dump(train_datagen, handle)


with open('config/trained_datagen.pickle', 'rb') as handle:
    train_datagen = pickle.load(handle)
