import os
import pandas as pd
from mole.data import DataLoader
from mole.pipeline import load_pipeline
from sklearn.metrics import mean_squared_error

root_path = ""


if __name__ == "__main__":
    # Test Data
    path_to_file = os.path.join(root_path, '../../data/logD/scaffold/test.csv')
    dataloader = DataLoader()
    test_smiles = dataloader.load_smiles(path_to_file)
    test_y = dataloader.rest.squeeze()

    # Test Data Augmented
    path_to_file = os.path.join(root_path, '../../data/logD/scaffold/test_taut.csv')
    dataloader = DataLoader()
    test_smiles_aug = dataloader.load_smiles(path_to_file)
    test_y_aug = dataloader.rest.squeeze()

    # Model #1
    pipe_stand = load_pipeline("../../pipelines/logD_Chemprop_NoTaut_202205130921")
    pipe_aug = load_pipeline("../../pipelines/logD_Chemprop_Taut_202205131004")

    print("Inference\n")
    print("\t1/4 ...")
    stand_model_stand_data = pipe_stand.predict(test_smiles)
    print("\t2/4 ...")
    stand_model_aug_data = pipe_stand.predict(test_smiles_aug)
    print("\t3/4 ...")
    aug_model_stand_data = pipe_aug.predict(test_smiles)
    print("\t4/4 ...")
    aug_model_aug_data = pipe_aug.predict(test_smiles_aug)

    rmse_SS = mean_squared_error(test_y, stand_model_stand_data, squared=False)
    rmse_SA = mean_squared_error(test_y_aug, stand_model_aug_data, squared=False)

    rmse_AS = mean_squared_error(test_y, aug_model_stand_data, squared=False)
    rmse_AA = mean_squared_error(test_y_aug, aug_model_aug_data, squared=False)


    rmses_df = pd.DataFrame({"Standard Model": [rmse_SS, rmse_SA,],
                             "Tautomer Model": [rmse_AS, rmse_AA,],},
                            index=["Test Data", "Tautomer Test Data"])

    print(rmses_df.T)

    # g = sns.heatmap(rmses_df.T, annot=True, vmin=0.55, vmax=1)
    # g.set_title("Chemprop MPNN \n logD")
    # plt.savefig('results/performance/logD_Chemprop_1000epochs.png', bbox_inches='tight')
    # plt.show()



    ########################################################################################
    # # Model SCAFFOLD
    # path_to_file = os.path.join(root_path, 'data/logD/scaffold/test.csv')
    # dataloader = DataLoader()
    # test_smiles = dataloader.load_smiles(path_to_file)
    # test_y = dataloader.rest.squeeze()
    #
    # pipe = load_pipeline("pipelines/logD_Chemprop_NoTaut_202205130921")
    #
    # yhat = pipe.predict(test_smiles)
    # rmse_scaffold = mean_squared_error(test_y, yhat, squared=False)
    #
    # print("RMSE Scaffold Data (Scaffold Model) = ", rmse_scaffold)
    #
    #
    # # Model MULTISPLIT
    # path_to_file = os.path.join(root_path, 'data/logD/multi/test_union.csv')
    # dataloader = DataLoader()
    # test_smiles_union = dataloader.load_smiles(path_to_file)
    # test_y_union = dataloader.rest.squeeze()
    #
    # path_to_file = os.path.join(root_path, 'data/logD/multi/test_intersec.csv')
    # dataloader = DataLoader()
    # test_smiles_intersec = dataloader.load_smiles(path_to_file)
    # test_y_intersec = dataloader.rest.squeeze()
    #
    # pipe = load_pipeline("pipelines/logD_Chemprop_NoTaut_Multi_202205130942")
    #
    # yhat_union = pipe.predict(test_smiles_union)
    # yhat_intersec = pipe.predict(test_smiles_intersec)
    #
    # rmse_union = mean_squared_error(test_y_union, yhat_union, squared=False)
    # rmse_intersec = mean_squared_error(test_y_intersec, yhat_intersec, squared=False)
    #
    # print("RMSE MultiSplit Union Data (MultiSplit Model) = ", rmse_union)
    # print("RMSE MultiSplit Intersection (MultiSplit Model) = ", rmse_intersec)

    # rmses_df = pd.DataFrame({"Standard Model": [rmse_SS, rmse_SA,],
    #                          "Tautomer Model": [rmse_AS, rmse_AA,],},
    #                         index=["Test Data", "Tautomer Test Data"])
    #
    # print(rmses_df)
    #
    # g = sns.heatmap(rmses_df.T, annot=True, vmin=0.4, vmax=1)
    # g.set_title("Chemprop MPNN \n logD")
    # plt.savefig('results/performance/logD_Chemprop.png', bbox_inches='tight')
    # plt.show()












