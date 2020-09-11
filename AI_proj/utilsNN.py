import numpy as np


def split_train_and_test(num_examples, test_percentage):
    """
    Splits a number into training and test randomly
    :param num_examples: int of the number of examples
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples * test_percentage))
    # Train and validation indexes
    train_val_idx = all_samples_idx[0:len(all_samples_idx) - test_examples]
    test_idx = all_samples_idx[len(all_samples_idx) - test_examples:len(all_samples_idx)]

    return [train_val_idx, test_idx]


def split_train_validation_and_test(num_examples, val_percentage, test_percentage):
    """
    Splits a number into training, validation, and test randomly
    :param num_examples: int of the number of examples
    :param val_percentage: int of the percentage desired for validation
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples * test_percentage))
    val_examples = int(np.ceil(num_examples * val_percentage))
    # Train and validation indexes
    train_idx = all_samples_idx[0:len(all_samples_idx) - test_examples - val_examples]
    val_idx = all_samples_idx[len(all_samples_idx) - test_examples - val_examples:len(all_samples_idx) - test_examples]
    test_idx = all_samples_idx[len(all_samples_idx) - test_examples:]
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    return [train_idx, val_idx, test_idx]
