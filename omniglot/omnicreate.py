import argparse
import numpy as np
import os
import pickle

from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, type=str, default=None)
args = parser.parse_args()
assert (args.data_dir is not None) and (os.path.isdir(args.data_dir))


def load():
    # load data
    file = os.path.join(args.data_dir, 'chardata.mat')
    data = loadmat(file)

    # data is in train/test split so read separately
    train_images = data['data'].astype(np.float32).T
    train_alphabets = np.argmax(data['target'].astype(np.float32).T, axis=1)
    train_characters = data['targetchar'].astype(np.float32)

    test_images = data['testdata'].astype(np.float32).T
    test_alphabets = np.argmax(data['testtarget'].astype(np.float32).T, axis=1)
    test_characters = data['testtargetchar'].astype(np.float32)

    # combine train and test data
    images = np.concatenate([train_images, test_images], axis=0)
    alphabets = np.concatenate([train_alphabets, test_alphabets], axis=0)
    characters = np.concatenate([np.ravel(train_characters),
                                 np.ravel(test_characters)], axis=0)
    data = (images, alphabets, characters)

    return data


def modify(data):
    # We don't care about alphabets, so combine all alphabets
    # into a single character ID.
    # First collect all unique (alphabet, character) pairs.
    images, alphabets, characters = data
    unique_alphabet_character_pairs = list(set(zip(alphabets, characters)))

    # Now assign each pair an ID
    ids = np.asarray([unique_alphabet_character_pairs.index((alphabet, character))
                      for (alphabet, character) in zip(alphabets, characters)])

    # Now split into train(1200)/val(323)/test(100) by character
    train_images = images[ids < 1200]
    train_labels = ids[ids < 1200]
    val_images = images[(1200 <= ids) * (ids < 1523)]
    val_labels = ids[(1200 <= ids) * (ids < 1523)]
    test_images = images[1523 <= ids]
    test_labels = ids[1523 <= ids]

    split_data = (train_images, train_labels, val_images,
                  val_labels, test_images, test_labels)

    return split_data


def save(data):
    savepath = os.path.join(args.data_dir, 'train_val_test_split.pkl')
    with open(savepath, 'wb') as file:
        pickle.dump(data, file)


def main():
    data = load()
    modified_data = modify(data)
    save(modified_data)

if __name__ == '__main__':
    main()