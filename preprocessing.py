import re
import pandas as pd

# Columns the model will READ (inputs / features)
TEXT_COLUMNS = ["subject", "body"]

# Columns the model will LEARN to predict (labels)
LABEL_COLUMNS = ["type", "queue", "priority"]


def clean_text(text):
    """Make one ticket's text simple and consistent."""
    text = str(text).lower()                           # lowercase everything
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", text)   # remove placeholders like <tel_num> or [acc_num]
    text = re.sub(r"\s+", " ", text)                   # turn many spaces/new lines into one space
    return text.strip()                                # remove spaces at the start and end


def load_and_prepare(path="data/tickets.csv"):
    """Read the CSV and return a clean table: text + the 3 label columns."""
    df = pd.read_csv(path)

    # Empty subject/body cells become empty text instead of NaN
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")

    # Join subject and body into one text column and clean it
    df["text"] = (df["subject"] + " " + df["body"]).apply(clean_text)

    # Drop tickets that have no text at all, or a missing label
    df = df[df["text"] != ""]
    df = df.dropna(subset=LABEL_COLUMNS)

    df = df.reset_index(drop=True)
    return df[["text"] + LABEL_COLUMNS]


# This part only runs when you run this file directly (a quick test)
if __name__ == "__main__":
    data = load_and_prepare()
    print("Rows after cleaning:", len(data))
    print("Columns:", list(data.columns))
    print("Empty texts left:", (data["text"] == "").sum())
    print("Sample text:", data.loc[0, "text"][:200])

