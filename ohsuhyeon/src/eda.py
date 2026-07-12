import ast
import collections
import pandas as pd
from config import TRAIN_CSV, TRAIN_IMG_DIR

def main():
    df = pd.read_csv(TRAIN_CSV)
    print(f"rows: {len(df)}")

    answers = df["Answer"].apply(ast.literal_eval)
    valid = answers.apply(lambda a: sorted(a) == [1, 2, 3, 4])
    print(f"rows with a valid 1..4 permutation answer: {valid.sum()} / {len(df)}")

    counts = collections.Counter(tuple(a) for a in answers)
    print(f"distinct permutation classes seen: {len(counts)} / 24")
    majority_class, majority_n = counts.most_common(1)[0]
    print(f"majority class {majority_class}: {majority_n} rows "
          f"({majority_n / len(df):.1%}) -> naive 'always predict this' floor")

    print(f"No_ordering distribution: {df['No_ordering'].value_counts().to_dict()}")

    missing_images = 0
    for _, row in df.iterrows():
        folder = TRAIN_IMG_DIR / row["Id"]
        for col in ("Input_1", "Input_2", "Input_3", "Input_4"):
            if not (folder / row[col]).exists():
                missing_images += 1
    print(f"missing image files: {missing_images}")

if __name__ == "__main__":
    main()
