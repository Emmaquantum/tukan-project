from pathlib import Path
import pandas as pd

def data_call():
    path = str(Path(__file__).parent.resolve())
    path_data = path + "\\banxico_auction_rates_monthly.csv"
    df = pd.read_csv(path_data, sep=",")
    return df