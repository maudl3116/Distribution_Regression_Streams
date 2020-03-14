import sys
import numpy as np
import utils
import experiments
import signature_features
from data_generators import ellipsis
import pickle

def exp1(N_MC,noise_pos,N_bags=100, N_items=15, tspan=np.linspace(0,2*np.pi,100),spec_param = {'a':[1.,3.], 'b':[1.,3.]}):

    # in this experiment, we evaluate the robustness of two regression models to the position noise of the ellipsis.
    example = ellipsis.Ellipsis()

    df_train_r2 = pd.DataFrame(index=N_MC * ['train_r2'], columns=noise_pos)
    df_train_rmse = pd.DataFrame(index=N_MC * ['train_rmse'], columns=noise_pos)
    df_test_r2 = pd.DataFrame(index=N_MC * ['test_r2'], columns=noise_pos)
    df_test_rmse = pd.DataFrame(index=N_MC * ['test_rmse'], columns=noise_pos)

    for i in range(N_MC):
        for j,param in enumerate(noise_pos):

            ''' GENERATE DATA '''
            example.generate_data(N_bags=N_bags, N_items=N_items, t_span=tspan, spec_param=spec_param, stdv_pos=param,
                          stdv_noise=0.3)
            example.e_ang()

            ''' PREPARE DATA FOR REGRESSION '''
            data_scaled, y_scaled, train_indices, test_indices = utils.split_standardize(example.labels, example.paths,
                                                                                     standardized=True,
                                                                                     method='stratify')


            ''' GP NAIVE '''
            dim_1 = np.array(np.array(data_scaled)[:, :, :, 0])
            dim_2 = np.array(np.array(data_scaled)[:, :, :, 1])
            input_ = np.concatenate([dim_1, dim_2], axis=2)

            train_RBF_rmse, train_RBF_r2, test_RBF_rmse, test_RBF_r2 = experiments.naive_experiment(input_, y_scaled, train_indices, test_indices,
                                                               param_init=[5, 1, 0])


            ''' GP SIG '''

            expected_pathwise_sig = signature_features.scaled_expected_sig([e.copy() for e in data_scaled], 6, M=1000,
                                                                       a=1, ilya_rescale=True, return_norms=False)
            K_precomputed = experiments.precompute_K(expected_pathwise_sig)
            train_Sig_rmse, train_Sig_r2, test_Sig_rmse, test_Sig_r2 = experiments.experiment_precomputed(K_precomputed, y_scaled, train_indices, test_indices, RBF=True,
                                                         plot=False)


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
    pickle.dump(df,open( "exp1.obj", "wb" ))


if __name__ == "__main__":
    # this will be used later to store results of experiment
    main(sys.argv[0],sys.argv[1])