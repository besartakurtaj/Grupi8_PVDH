import sys
import pandas as pd

def to_title_with_spaces(col: str) -> str:
    return " ".join(part.capitalize() for part in str(col).replace("_", " ").split())

def titlecase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_title_with_spaces(c) for c in df.columns]
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python titlecase_columns_spaces.py <input.csv> <output.csv>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(in_path)
    df = titlecase_columns(df)
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
