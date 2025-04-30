import pandas as pd
from sklearn.model_selection import train_test_split

class ProcessDataset:
    
    @staticmethod
    def split_dataset(path:str) -> None:
        print("Procesando dataset...")
        df = pd.read_csv(path)
        array = path.split("/")[:2]
        path = "/".join(array)
        print(path)
        train_df, test_df = train_test_split(df,test_size=0.20, random_state=50)

        train_path = path + "/train/NFLX_train.csv"
        test_path = path + "/test/NFLX_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False) 
        print("Dataset procesado") 