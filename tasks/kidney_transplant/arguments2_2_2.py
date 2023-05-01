## model parameters
# check below for xgboost parameter
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
# to-do: dictionary sanity check

import numpy as np



"""model choice"""
'''XGBRegressor: Need RegressorChain'''
model='xgboost.XGBRegressor'

'''SVR: Need RegressorChain'''
# model='sklearn.svm.SVR'

'''RandomForestRegressor: RegressorChain is not necessary'''
# model='sklearn.ensemble.RandomForestRegressor'

'''ElasticNet: RegressorChain is not necessary'''
# model='sklearn.linear_model.ElasticNet'

'''Simple Linear Regression'''
# model='sklearn.linear_model.LinearRegression'


"""mode setting"""
'''If both single and next are True, then given_day is fixed to (end_day - 1).'''
'''Either single or next must be True.'''
'''single:False & next:True (regressorchain) mode is not stable'''
target_is_single = True
target_is_next = True


''''predict either dose or fk level'''
target_is_dose = False
""" None, 'random' or 'grid' """
param_searching_method = None

'''train or test'''
phase = 'test'

'''ETC'''
n_jobs = 8 # -1, 8, 16 ~ ?
seed = 42


config1 = dict(
    runner='scikit_learn2_2_2',
    ## data related part
    data_loader='tasks.kidney_transplant.Dataset_KT2_2_2',
    
    ## training related part
    phase=phase,
    validation=False,
    batch_size=64,
#     learning_rate=0.01,
    n_jobs=n_jobs,
    logging=True,
    cv_fold=0,  # not use cross-validation if cv_fold=0
    test_size=0.25,
    param_searching_method=param_searching_method,
    seed=seed,
    target_is_single=target_is_single,
    given_day=10, # day: 1 ~ 11. If either single or next is True, the day goes 1 to (end_day-1)
    end_day=11, # input data period: given_day ~ end_day - 1, given_day < end_day
    target_is_next=target_is_next, # predict either next or end day
    error_range = 0.25,

    
    ## model part
    checkpoint='./checkpoint/task2_2_2/model_',
    model_type = 'regressor', # regressor or classifier
    feature_names = 'feature_names_in_',
)

if model == 'xgboost.XGBRegressor':
    config2_0 = dict(
         dataset="./data/kt_registry/processed/processed2(ordinal).xlsx",
         testset_dose="./data/kt_registry/processed/processed2_test_dose(ordinal)",
         testset_fk="./data/kt_registry/processed/processed2_test_fk(ordinal)",
         feature_importances='feature_importances_',
         model=model,
         scaling=False,
         feature_names = 'feature_names',
         training_callback='xgboost.callback.EarlyStopping',
         callback_kwargs=dict(
             rounds=50,
             save_best=True,
         ),
    )
    if target_is_dose:
        config2 = dict(
            model_params= dict(
                booster='gbtree',
                verbosity=1,
                n_jobs=n_jobs,
                
                learning_rate=0.03,
                n_estimators=200,
                max_depth=2,
                min_child_weight=2.25,
                gamma=0.1,
                subsample=0.8,
                reg_lambda=0.5,
                reg_alpha=1,
                colsample_bylevel=1,
                colsample_bynode=1,
                colsample_bytree=1,

                objective="reg:squarederror",
                eval_metric=["rmse"],

                seed=seed,
            ),
        )
    else:
        config2 = dict(
            model_params= dict(
                booster='gbtree',
                verbosity=1,
                n_jobs=n_jobs,

                learning_rate=0.02,
                n_estimators=600,
                max_depth=2,
                min_child_weight=0.1,
                gamma=14,
                subsample=0.8,
                reg_lambda=1,
                reg_alpha=1,
                colsample_bylevel=0.7,
                colsample_bynode=0.7,
                colsample_bytree=1,

                objective="reg:squarederror",
                eval_metric=["rmse"],

                random_state=seed,
            ),
        )
    config2 = {**config2_0, **config2}
            
if model == 'xgboost.XGBRegressor':
    if target_is_dose:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    booster=['gbtree'],

                    learning_rate=[0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8],
                    n_estimators=[100, 150, 200, 250, 300, 400, 600, 800, 1200, 1600, 2400 , 3200, 6400],
                    gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    max_depth=[2, 3, 4, 5, 6, 7, 8, 16, 32, 64, None],
                    min_child_weight=[0.625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16],
                    subsample=[0.5, 0.6, 0.7, 0.8, 1],
                    reg_lambda=[0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16],
                    reg_alpha=[0,  0.001, 0.01, 0.1, 1, 10],
                    colsample_bytree=[0.6, 0.7, 0.8, 0.9,  1], 
                    colsample_bylevel=[0.6, 0.7, 0.8, 0.9,  1], 
                    colsample_bynode=[0.6, 0.7, 0.8, 0.9,  1],
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    booster=['gbtree'], # consider 'dart'

                    learning_rate=[0.03],
                    n_estimators=[100, 200, 300],
                    gamma=[0.1, 1, 10],
                    max_depth=[2, 3],
                    min_child_weight=[2, 2.25, 2.5],
                    subsample=[0.8],
                    reg_lambda=[0.5, 1, 2],
                    reg_alpha=[0.1, 1, 10],
#                     colsample_bytree=[1], 
#                     colsample_bylevel=[1], 
#                     colsample_bynode=[1],
                ),
            )
        else:
            temp = dict()
    else:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    booster=['gbtree'],

                    learning_rate=[0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8],
                    n_estimators=[100, 150, 200, 250, 300, 400, 600, 800, 1200, 1600, 2400 , 3200, 6400],
                    gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    max_depth=[2, 3, 4, 5, 6, 7, 8, 16, 32, 64, None],
                    min_child_weight=[0.625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16],
                    subsample=[0.5, 0.6, 0.7, 0.8, 1],
                    reg_lambda=[0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16],
                    reg_alpha=[0,  0.001, 0.01, 0.1, 1, 10],
                    colsample_bytree=[0.6, 0.7, 0.8, 0.9,  1], 
                    colsample_bylevel=[0.6, 0.7, 0.8, 0.9,  1], 
                    colsample_bynode=[0.6, 0.7, 0.8, 0.9,  1],
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    booster=['gbtree'],

                    learning_rate=[0.02],
                    n_estimators=[500, 600, 700, 800],
                    gamma=[12, 14, 16],
                    max_depth=[2, 3],
                    min_child_weight=[0.1, 1],
                    subsample=[0.8],
                    reg_lambda=[1, 2],
                    reg_alpha=[1, 1.5],
                    colsample_bytree=[1],
                    colsample_bylevel=[0.7],
                    colsample_bynode=[0.7],
                ),
            )
        else:
            temp = dict()
    config2 = {**config2, **temp}
        

if model == 'sklearn.ensemble.RandomForestRegressor':
    config2_0 = dict(
        dataset="./data/kt_registry/processed/processed2(ordinal).xlsx",
        testset_dose="./data/kt_registry/processed/processed2_test_dose(ordinal)",
        testset_fk="./data/kt_registry/processed/processed2_test_fk(ordinal)",
        feature_importances='feature_importances_',
        model=model,
        scaling=False,
    )
    if target_is_dose:
        config2 = dict(
            model_params= dict(
                n_jobs=n_jobs,
                bootstrap=True,
                criterion='squared_error',
                max_depth=24,
                min_samples_split=4,
                min_samples_leaf=3,
                max_features=None,
                n_estimators=150,
                random_state=seed,
            ),
#             # random search
#             search_params=dict(
#                 bootstrap=[True, False],
#                 n_estimators=[100, 150, 200, 250, 300, 400, 600, 800, 1200, 1600, 2400 , 3200],
#                 max_depth=[2, 3, 4, 5, 6, 8, 12, 16, 32, 64, None],
#                 min_samples_split=[2, 5, 10],
#                 min_samples_leaf=[1, 2, 4],
#                 max_features=['sqrt', 'log2', None],
#             ),
            # grid search
            search_params=dict(
                bootstrap=[True, False],
                n_estimators=[100, 150, 200, 250, 300],
                max_depth=[16, 24, 32, 40, 48, None],
                min_samples_split=[2, 3, 4],
                min_samples_leaf=[1, 2, 3],
                max_features=['sqrt', 'log2', None],
            ),
        )
    else:
        config2 = dict(
            model_params= dict(
                n_jobs=n_jobs,
                bootstrap=True,
                criterion='squared_error',
                max_depth=32,
                min_samples_split=14,
                min_samples_leaf=2,
                max_features='sqrt',
                n_estimators=1000,
                random_state=seed,
            ),
#             # random search
#             search_params=dict(
#                 bootstrap=[True, False],
#                 n_estimators=[100, 150, 200, 250, 300, 400, 600, 800, 1200, 1600, 2400 , 3200],
#                 max_depth=[2, 3, 4, 5, 6, 8, 12, 16, 32, 64, None],
#                 min_samples_split=[2, 5, 10],
#                 min_samples_leaf=[1, 2, 4],
#                 max_features=['sqrt', 'log2', None],
#             ),
            # grid search
            search_params=dict(
                bootstrap=[True, False],
                n_estimators=[800, 1000, 1200, 1400, 1600],
                max_depth=[32, 48, 64, 80, 96, None],
                min_samples_split=[6, 8, 10, 12, 14],
                min_samples_leaf=[1, 2, 3],
                max_features=['sqrt', 'log2', None],
            ),
        )
    config2 = {**config2_0, **config2}

        
if model == 'sklearn.svm.SVR':
    config2_0 = dict(
        dataset="./data/kt_registry/processed/processed2(onehot).xlsx",
        testset_dose="./data/kt_registry/processed/processed2_test_dose(onehot)",
        testset_fk="./data/kt_registry/processed/processed2_test_fk(onehot)",
        feature_importances='dual_coef_',
        model=model,
        scaling=True,
    )
    if target_is_dose:
        config2 = dict(
            model_params= dict(
                kernel='rbf', 
                gamma='scale', 
                tol=1e-3, 
                C=1, 
                epsilon=0.1,
                max_iter=-1,
            ),
        )
    else:
        config2 = dict(
            model_params= dict(
                kernel='rbf', 
                gamma='scale', 
                tol=1e-3, 
                C=1, 
                epsilon=0.1,
                max_iter=-1,
            ),
        )
    config2 = {**config2_0, **config2}
    
if model == 'sklearn.svm.SVR':
    if target_is_dose:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    kernel=['linear', 'poly', 'rbf', 'sigmoid'],
#                     gamma=['auto', 'scale'],
#                     tol=[1e-4, 1e-3, 1e-2], 
#                     C=np.logspace(-2,2,5), 
#                     epsilon=np.logspace(-3,1,5),
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    kernel=['linear', 'rbf'],
#                     gamma=['auto', 'scale'],
                    tol=[1e-4, 1e-3, 1e-2], 
                    C=np.logspace(-2,2,10), 
                    epsilon=np.logspace(-3,1,10),
                ),
            )
        else:
            temp = dict()
    else:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    kernel=['linear', 'poly', 'rbf', 'sigmoid'],
#                     gamma=['auto', 'scale'],
#                     tol=[1e-4, 1e-3, 1e-2], 
#                     C=np.logspace(-2,2,5), 
#                     epsilon=np.logspace(-3,1,5),
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    kernel=['linear', 'rbf'],
#                     gamma=['auto', 'scale'],
                    tol=[1e-4, 1e-3, 1e-2], 
                    C=np.logspace(-2,2,10), 
                    epsilon=np.logspace(-3,1,10),
                ),
            )
        else:
            temp = dict()
    config2 = {**config2, **temp}
                
                
if model == 'sklearn.linear_model.ElasticNet':
    config2_0 = dict(
        dataset="./data/kt_registry/processed/processed2(onehot).xlsx",
        testset_dose="./data/kt_registry/processed/processed2_test_dose(onehot)",
        testset_fk="./data/kt_registry/processed/processed2_test_fk(onehot)",
        feature_importances='coef_',
        model=model,
        scaling=True,
    )
    if target_is_dose:
        config2 = dict(
            model_params=dict(
                alpha=0.03, 
                l1_ratio=0.9, 
                tol=0.0064,
#                 max_iter=1000, 
            ),
        )
    else:
        config2 = dict(
            model_params=dict(
                alpha=0.05, 
                l1_ratio=1, 
                tol=0.0032,
#                 max_iter=1000, 
            ),
        )
    config2 = {**config2_0, **config2}
                
if model == 'sklearn.linear_model.ElasticNet':
    if target_is_dose:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    alpha=[1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 10, 16, 32, 64], 
                    l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                    tol=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    alpha=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05], 
                    l1_ratio=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                    tol=[1e-5, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.4096],
                ),
            )
        else:
            temp = dict()
    else:
        if param_searching_method == 'random':
            temp = dict(
                search_params=dict(
                    alpha=[1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 10, 16, 32, 64], 
                    l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                    tol=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                ),
            )
        elif param_searching_method == 'grid':
            temp = dict(
                search_params=dict(
                    alpha=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1], 
                    l1_ratio=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                    tol=[1e-5, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.4096],
                ),
            )
        else:
            temp = dict()
    config2 = {**config2, **temp}
                
    
if model == 'sklearn.linear_model.LinearRegression':
    config2 = dict(
        dataset="./data/kt_registry/processed/processed2(onehot).xlsx",
        testset_dose="./data/kt_registry/processed/processed2_test_dose(onehot)",
        testset_fk="./data/kt_registry/processed/processed2_test_fk(onehot)",
        feature_importances='coef_',
        model=model,
        scaling=True,
        model_params= dict(
        ),
        search_params=dict(
        ),
    )

# mlp_params = dict(
#     learning_rate_init=0.01,
#     max_iter=100
# )
    
if 'onehot' in config2['dataset'].lower():
    if target_is_dose:
        if target_is_single and target_is_next:
            config3 = dict(
                inputs = [],
                float_columns = [],
                target_name = 'dose',
            )
        elif (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            config3 = dict(
                inputs = [],
                target_name = 'dose',
            )
    else:
        if target_is_single and target_is_next:
            config3 = dict(
                inputs = [],
                float_columns = [],
                target_name = 'fk',
            )
        elif (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            config3 = dict(
                    inputs = [],
                    target_name = 'fk',
                )
            
elif 'ordinal' in config2['dataset'].lower():
    if target_is_dose:
        if target_is_single and target_is_next:
            config3 = dict(
                inputs = [],
                float_columns = [],
                target_name = 'dose',
            )
        elif (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            config3 = dict(
                inputs = [],
                target_name = 'dose',
            )
    else:
        if target_is_single and target_is_next:
            config3 = dict(
                inputs = [],
                float_columns = [],
                target_name = 'fk',
            )
        elif (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            config3 = dict(
                inputs = [],
                target_name = 'fk',
            )


if param_searching_method:
    config_temp = dict(
        phase = 'train',
        cv_fold=5,
    )
    config1.update(config_temp)
else:
    if phase == 'test':
        config_temp = dict(
            cv_fold=0,
        )
        config1.update(config_temp)
        
if target_is_single and target_is_next:
        config_temp = dict(
            given_day=config1['end_day']-1,
        )
        config1.update(config_temp)
            

config = {**config1, **config2, **config3}