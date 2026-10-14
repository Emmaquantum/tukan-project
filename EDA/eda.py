import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Data.data import data_call

class Clean:
    def __init__(self,df):
        self.df = df.copy()
        self.cols = df.columns.tolist()
        self.num_cols = self.df.select_dtypes(include="number").columns.tolist()
        self.str_cols = self.df.select_dtypes(include=["object", "string"]).columns.tolist()

    def delete_rows(self):
        return self.df.dropna(subset=self.cols)

    def delete_duplicate(self, col):
        return self.df.drop_duplicates(subset=col)

    def text_clean(self):
        for col in self.str_cols:
            self.df[col] = self.df[col].str.lower().str.strip()
        return self.df

    def clean(self):
        self.df = self.delete_rows()
        self.df= self.text_clean()

        return self.df

    def separate(self):
        self.df["vector"] = self.df.INSTRUMENT.apply(lambda x: x.split(","))
        self.df["INSTRUMENT"] = self.df.vector.apply(lambda x: x[0])
        self.df["Contract-time"] = self.df.vector.apply(lambda x: x[1])

        self.cols.append("Contract-time")
        return self.df

    def uniques(self):
        self.cols.remove("vector")
        return {col: len(self.df[col].unique()) for col in self.cols}

    def discriminator(self):
        dias = self.df["Contract-time"].str.extract(r"(\d+)")[0].astype(int)
        return {d: g.sort_values("DATE") for d, g in self.df.groupby(dias)}

    def plot_serietimes(self, df, label=None):
        plt.plot(df.DATE, df.VALUE_PCT, linewidth=1.5, label=label)
