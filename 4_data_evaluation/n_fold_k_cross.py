from sklearn.model_selection import KFold
import numpy as np

def n_fold_k_cross_validation(dataset, n_splits=5):
    """
    Perform n-fold k-cross validation on the given dataset.

    Parameters:
    dataset (array-like): The dataset to be split.
    n_splits (int): Number of folds. Must be at least 2.

    Returns:
    list: A list of tuples containing train and test indices for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for train_index, test_index in kf.split(dataset):
        splits.append((train_index, test_index))

    return splits

# Example usage:
if __name__ == "__main__":
    # Example dataset
    dataset = np.arange(100)
    splits = n_fold_k_cross_validation(dataset, n_splits=5)

    for i, (train_index, test_index) in enumerate(splits):
        print(f"Fold {i+1}")
        print("Train indices:", train_index)
        print("Test indices:", test_index)
        print()