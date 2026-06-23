import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def balance_classes(df: pd.DataFrame, max_per_class: int = 2000) -> pd.DataFrame:
    """
    Cap each class at max_per_class to prevent dominant classes
    from overwhelming the minority classes during training.
    """
    return (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), max_per_class), random_state=42))
          .reset_index(drop=True)
    )


def split_dataset(df: pd.DataFrame) -> DatasetDict:
    """Stratified 80/10/10 train/val/test split."""
    def manual_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
        n_rows = len(shuffled)

        if n_rows < 3:
            return shuffled, shuffled.iloc[0:0], shuffled.iloc[0:0]

        train_size = max(1, int(round(n_rows * 0.8)))
        val_size = max(1, int(round(n_rows * 0.1)))
        test_size = n_rows - train_size - val_size

        if test_size < 1:
            deficit = 1 - test_size
            reducible_train = max(0, train_size - 1)
            take_from_train = min(deficit, reducible_train)
            train_size -= take_from_train
            deficit -= take_from_train

            reducible_val = max(0, val_size - 1)
            take_from_val = min(deficit, reducible_val)
            val_size -= take_from_val
            deficit -= take_from_val

            test_size = n_rows - train_size - val_size

        train_df = shuffled.iloc[:train_size].reset_index(drop=True)
        val_df = shuffled.iloc[train_size:train_size + val_size].reset_index(drop=True)
        test_df = shuffled.iloc[train_size + val_size:].reset_index(drop=True)
        return train_df, val_df, test_df

    try:
        label_counts = df["label"].value_counts()
        can_stratify = len(label_counts) > 1 and label_counts.min() >= 2

        split_kwargs = {"random_state": 42}
        if can_stratify:
            split_kwargs["stratify"] = df["label"]

        train_df, temp_df = train_test_split(df, test_size=0.2, **split_kwargs)

        temp_counts = temp_df["label"].value_counts()
        can_stratify_temp = len(temp_counts) > 1 and temp_counts.min() >= 2

        split_kwargs_temp = {"random_state": 42}
        if can_stratify_temp:
            split_kwargs_temp["stratify"] = temp_df["label"]

        val_df, test_df = train_test_split(temp_df, test_size=0.5, **split_kwargs_temp)
    except ValueError:
        train_df, val_df, test_df = manual_split(df)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return DatasetDict({
        "train":      Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df,   preserve_index=False),
        "test":       Dataset.from_pandas(test_df,  preserve_index=False),
    })