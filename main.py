from src.library.facade import AImodel
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(path:str) -> None:
  df = pd.read_csv(path)
  array = path.split("/")[:2]
  path = "/".join(array)
  print(path)
  train_df, temp_df = train_test_split(df,test_size=0.30, random_state=50)
  val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=50)

  train_path = path + "/train/NFLX_train.csv"
  val_path = path + "/val/NFLX_val.csv"
  test_path = path + "/test/NFLX_test.csv"

  train_df.to_csv(train_path, index=False)
  val_df.to_csv(val_path, index=False)
  test_df.to_csv(test_path, index=False)



if __name__ == "__main__":
   #split_dataset("./dataset/NFLX_dataset.csv")
   model = AImodel("sgd")
   trainDic = {
      "metric": "rmse",
      "cv": 5,
      "jobs": 1,
      "train_score": True,
      "epoch": 10000
   }

   model.train("./dataset/train/NFLX_train.csv", trainDic)
   #model.predic("./dataset/test/NFLX_test.csv")
   model.save("SGD") 
   #lineal.load("")
