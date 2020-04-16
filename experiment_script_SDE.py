import sys
import numpy as np
import utils
import experiments
import signature_features
from data_generators import SDE
import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def exp(N_MC,N_bags=100, N_items=30, tspan=np.linspace(0,1,200),spec_param = {'theta_1':[0,2],'Y0':[0.,0.]},device=torch.device('cuda')):

    # in this experiment, we evaluate the robustness of two regression models to the position noise of the ellipsis.

    params = N_bags

    df_train_RBF_r2 = pd.DataFrame(index=N_MC * ['train_RBF_r2'], columns=params)
    df_train_RBF_rmse = pd.DataFrame(index=N_MC * ['train_RBF_rmse'], columns=params)
    df_test_RBF_r2 = pd.DataFrame(index=N_MC * ['test_RBF_r2'], columns=params)
    df_test_RBF_rmse = pd.DataFrame(index=N_MC * ['test_RBF_rmse'], columns=params)
    df_train_Sig_r2 = pd.DataFrame(index=N_MC * ['train_Sig_r2'], columns=params)
    df_train_Sig_rmse = pd.DataFrame(index=N_MC * ['train_Sig_rmse'], columns=params)
    df_test_Sig_r2 = pd.DataFrame(index=N_MC * ['test_Sig_r2'], columns=params)
    df_test_Sig_rmse = pd.DataFrame(index=N_MC * ['test_Sig_rmse'], columns=params)


    for i in range(N_MC):
        for j,param in enumerate(params):

            ''' GENERATE DATA '''
            example = SDE.sde(N_bags=param, N_items=N_items, t_span=tspan, spec_param=spec_param)

            example.generate_data()
            example.get_param()

            ''' PREPARE DATA FOR REGRESSION '''
            data_scaled, y_scaled, train_indices, test_indices = utils.split_standardize(example.labels, example.paths,
                                                                                     standardized=False,
                                                                                     method='stratify')

            X_aug = utils.add_dimension(data_scaled, add_time=False, lead_lag=0)

            ''' GP NAIVE '''
            input_ = X_aug
            N = np.array(input_).shape[3]
            input_ = np.concatenate([np.array(np.array(input_)[:, :, :, k]) for k in range(N)], axis=2)
            train_RBF_rmse, train_RBF_r2, test_RBF_rmse, test_RBF_r2 = experiments.naive_experiment(input_, y_scaled, train_indices, test_indices,ARD=False,
                                                               param_init=[10, 0, 0],device=device)


            ''' GP SIG '''
            sig_level = 3

            # Compute the expected signature
            expected_sig = signature_features.scaled_expected_sig([e.copy() for e in X_aug], sig_level)
            features = expected_sig

            ## potentially scale the features
            scaler = StandardScaler()
            to_fit = [features[i] for i in train_indices]
            scaler.fit(to_fit)
            features = scaler.transform(features)

            # Precompute the Gram matrix, as we do not optimize any parameter
            K_precomputed = experiments.precompute_K(features)

            # Train and Predict
            train_Sig_rmse, train_Sig_r2, test_Sig_rmse, test_Sig_r2 =experiments.experiments.experiment_precomputed(K_precomputed, y_scaled, train_indices, test_indices, RBF=False,
                                               plot=False,device=device)

            ''' STORE THE RESULTS '''
            df_train_RBF_rmse.iloc[i,j] = train_RBF_rmse
            df_train_RBF_r2.iloc[i,j] = train_RBF_r2
            df_test_RBF_rmse.iloc[i,j] = test_RBF_rmse
            df_test_RBF_r2.iloc[i,j] = test_RBF_r2
            df_train_Sig_rmse.iloc[i,j] = train_Sig_rmse
            df_train_Sig_r2.iloc[i,j] = train_Sig_r2
            df_test_Sig_rmse.iloc[i,j] = test_Sig_rmse
            df_test_Sig_r2.iloc[i,j] = test_Sig_r2

    df = pd.concat([df_train_RBF_rmse,df_train_RBF_r2,df_test_RBF_rmse,df_test_RBF_r2,df_train_Sig_rmse,df_train_Sig_r2,df_test_Sig_rmse,df_test_Sig_r2])
    pickle.dump(df,open( "exp_SDE_bags.obj", "wb" ))