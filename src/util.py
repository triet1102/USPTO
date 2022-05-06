import pandas as pd
import os


def preprocess_CPC(cpc_path):
    """Extract titles coresspond to code

    To download dataset
    wget https://www.cooperativepatentclassification.org/sites/default/files/cpc/bulk/CPCTitleList202205.zip

    :param cpc_path: Path of directory that contains cpc data
    """
    parsed = {x: [] for x in ["code", "title"]}
    for letter in "ABCDEFGHY":
        file = f"{cpc_path}/cpc-section-{letter}_20220501.txt"
        with open(file) as f:
            for line in f:
                vals = line.strip().split('\t')
                if len(vals[0]) == 3:
                    parsed["code"].append(vals[0])
                    parsed["title"].append(vals[2].lower())

    df = pd.DataFrame.from_dict(parsed)
    print(df.head())
    df.to_csv(f"{cpc_path}/../CPC_context.csv")


def main(cpc_path):
    preprocess_CPC(cpc_path)


if __name__ == "__main__":
    CPC_PATH = "../data/CPCTitleList"
    main(cpc_path=CPC_PATH)
