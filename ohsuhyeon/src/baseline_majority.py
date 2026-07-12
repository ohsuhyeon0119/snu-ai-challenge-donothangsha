import ast
import pandas as pd
from config import TRAIN_CSV
from validation import make_split, exact_match_accuracy

def main():
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = make_split(df)
    truths = [tuple(ast.literal_eval(a)) for a in val_df["Answer"]]
    preds = [(1, 2, 3, 4)] * len(val_df)
    acc = exact_match_accuracy(preds, truths)
    print(f"majority-class baseline val accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
