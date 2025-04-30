from src.library.Allibrary import AImodel
from src.helpers.ProcessDataset import ProcessDataset



if __name__ == "__main__":
   ProcessDataset.split_dataset("./dataset/NFLX_dataset.csv")
   model = AImodel("tf")
   trainDic = {
      "metric": "rmse",
      "cv": 15,
      "jobs": -1,
      "epoch": 10,
      "batch_size": 5,
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
   model.save_model("TF") 

   redy = AImodel("tf")
   redy.load_model("./TF/TF.keras")
   redy.test_model("./dataset/test/NFLX_test.csv")
