from src.library.facade import AImodel
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(path:str) -> None:
  df = pd.read_csv(path)
  array = path.split("/")[:2]
  path = "/".join(array)
  print(path)
  train_df, test_df = train_test_split(df,test_size=0.20, random_state=50)

  train_path = path + "/train/NFLX_train.csv"
  test_path = path + "/test/NFLX_test.csv"

  train_df.to_csv(train_path, index=False)
  test_df.to_csv(test_path, index=False)



if __name__ == "__main__":
   #split_dataset("./dataset/NFLX_dataset.csv")
   model = AImodel("sgd")
   trainDic = {
      "metric": "rmse",
      "cv": 15,
      "jobs": -1,
      "train_score": True,
      "epoch": 1000,
      "epsilon": 0.5,
      "penalty": "l2",
      "verbose": 1,
      "alpha": 0.0002,
      "n_iter_stop": 10,
      "random_state": 50,
      "early_stopping": True,
      "shuffle": True
   }

   model.train_model("./dataset/train/NFLX_train.csv", trainDic)
   model.test_model("./dataset/test/NFLX_test.csv")
   model.save_model("SGD") 

  #redy = AImodel("sgd")
  #redy.load_model("./SGD/SGD.pkl")
  #redy.test_model("./dataset/test/NFLX_test.csv")
