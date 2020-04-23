import sys
import numpy as np
import utils
import experiments
import signature_features
from data_generators import circuit
import pickle
import pandas as pd
import torch

def exp1(N_MC,subsample,N_bags=100, N_items=30, nb_periods=20,spec_param={'phi':[np.pi/8,np.pi/2],'em':[1,1],'omega':[5,5]}):

    # in this experiment, we evaluate the robustness of two regression models to the position noise of the ellipsis.


    df_train_RBF_r2 = pd.DataFrame(index=N_MC * ['train_RBF_r2'], columns=subsample)
    df_train_RBF_rmse = pd.DataFrame(index=N_MC * ['train_RBF_rmse'], columns=subsample)
    df_test_RBF_r2 = pd.DataFrame(index=N_MC * ['test_RBF_r2'], columns=subsample)
    df_test_RBF_rmse = pd.DataFrame(index=N_MC * ['test_RBF_rmse'], columns=subsample)
    df_train_Sig_r2 = pd.DataFrame(index=N_MC * ['train_Sig_r2'], columns=subsample)
    df_train_Sig_rmse = pd.DataFrame(index=N_MC * ['train_Sig_rmse'], columns=subsample)
    df_test_Sig_r2 = pd.DataFrame(index=N_MC * ['test_Sig_r2'], columns=subsample)
    df_test_Sig_rmse = pd.DataFrame(index=N_MC * ['test_Sig_rmse'], columns=subsample)

    for i in range(N_MC):
        for j,param in enumerate(subsample):

            ''' GENERATE DATA '''
            t_span = np.linspace(0, nb_periods * 2 * np.pi / spec_param['omega'][0], nb_periods * 100)
            # this is what changes
            nb_obs = nb_periods * param
            # generate data
            example = circuit.circuit(N_bags=N_bags, N_items=N_items, spec_param=spec_param,t_span=t_span,nb_obs=nb_obs)
            example.generate_data()
            example.get_phi()

            ''' PREPARE DATA FOR REGRESSION '''
            data_scaled, y_scaled, train_indices, test_indices = utils.split_standardize(example.labels, example.paths,
                                                                                         standardized=True,
                                                                                         method='stratify')

            ''' GP NAIVE '''
            input_ = data_scaled
            N = np.array(input_).shape[3]
            input_ = np.concatenate([np.array(np.array(input_)[:, :, :, k]) for k in range(N)], axis=2)

            train_RBF_rmse, train_RBF_r2, test_RBF_rmse, test_RBF_r2 = experiments.naive_experiment(input_, y_scaled,
                                                                                                    train_indices,
                                                                                                    test_indices,
                                                                                                    RBF_top=True,
                                                                                                    param_init=[15, 0,
                                                                                                                0, 0],
                                                                                                    device=torch.device(
                                                                                                        'cuda'),
                                                                                                    plot=True)

            ''' GP SIG '''

            expected_sig = signature_features.scaled_expected_sig([e.copy() for e in data_scaled], 6, M=1000,
                                                                  a=1, ilya_rescale=True, return_norms=False)

            K_precomputed = experiments.precompute_K(expected_sig)
            train_Sig_rmse, train_Sig_r2, test_Sig_rmse, test_Sig_r2 = experiments.experiment_precomputed(K_precomputed, y_scaled, train_indices, test_indices,
                                                            param_init=[0, 0, 0], RBF=True, plot=True)

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
    pickle.dump(df,open( "exp_phase_circuit.obj", "wb" ))

