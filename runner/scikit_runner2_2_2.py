from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from utils.reflection import get_function, get_class
import joblib
from utils.linegraph import linegraph_step
import numpy as np
import pandas as pd
import os


import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='default')

class Runner:
    def __init__(self, args):
        self.args = args
        self.given_day = self.args.get('given_day')

        self.MAE_record = []
        self.MSE_record = []
        self.RMSE_record = []
        self.underestimated_record = []
        self.overestimated_record = []
        self.max_E_record = []

        
    def load_model(self):
        if self.args.get('target_is_next'):
            model_location = f"{self.args.get('checkpoint')}({self.args.get('model')})_target_({self.args.get('target_name')}{self.given_day+1}_{self.given_day}days_given).pkl"
        else:
            model_location = f"{self.args.get('checkpoint')}({self.args.get('model')})_target_({self.args.get('target_name')}{self.args.get('end_day')}_{self.given_day}days_given).pkl"
        self.model = joblib.load(model_location)

        
    def save_model(self):
        if self.args.get('target_is_next'):
            model_location = f"{self.args.get('checkpoint')}({self.args.get('model')})_target_({self.args.get('target_name')}{self.given_day+1}_{self.given_day}days_given).pkl"
        else:
            model_location = f"{self.args.get('checkpoint')}({self.args.get('model')})_target_({self.args.get('target_name')}{self.args.get('end_day')}_{self.given_day}days_given).pkl"
        joblib.dump(self.model, model_location)

        
    def get_model(self):
        return get_class(self.args['model'])

    
    def get_dataset(self, train=True):
        if train:
            return get_function(self.args["data_loader"] + '.get_data')(self.args["dataset"], self.args.get('inputs'), self.given_day, self.args.get('end_day'), self.args.get('target_is_single'), self.args.get('target_is_next'), self.args.get('target_name'))
        else:
            if self.args.get('target_is_single'):
                temp = f"_target-{self.args.get('end_day')}_{self.args.get('given_day')}days_given.xlsx"
            elif self.args.get('target_is_next'):
                temp = f"_target-{self.args.get('given_day')+1}to{self.args.get('end_day')}.xlsx"  
            
            if self.args.get('target_name') == 'dose':
                return get_function(self.args["data_loader"] + '.get_test_data')(self.args["testset_dose"]+temp, self.args.get('target_is_single'), self.args.get('target_name'), self.args.get('given_day'))
            elif self.args.get('target_name') == 'fk':
                return get_function(self.args["data_loader"] + '.get_test_data')(self.args["testset_fk"]+temp, self.args.get('target_is_single'), self.args.get('target_name'), self.args.get('given_day'))
            
            
    def get_train_loader(self, data, use_scaler=True):
        inputs, targets = data
        data_balance = targets if self.args.get('data_balance') else None
        seed = self.args.get('seed')
        test_size = self.args.get('test_size') or 0.25
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets,
                                                            test_size=test_size,
                                                            stratify=data_balance,
                                                            random_state=seed)
        
        if self.args.get('scaling') and use_scaler:
            print("data is scaled")
            if (self.args.get('target_is_single') or self.args.get('target_is_next')) and not (self.args.get('target_is_single') and self.args.get('target_is_next')):
                ct = ColumnTransformer([('scaling', StandardScaler(), list(x_train.columns)), ], remainder='passthrough') # In future, beware of columns list for scaler
            elif self.args.get('target_is_single') and self.args.get('target_is_next'):
                ct = ColumnTransformer([('scaling', StandardScaler(), self.args.get('float_columns')), ], remainder='passthrough')
            
            train_index = x_train.index
            test_index = x_test.index
            columns = x_train.columns
        
            x_train = ct.fit_transform(x_train)
            x_test = ct.transform(x_test)

            x_train = pd.DataFrame(data=x_train, index=train_index, columns=columns)
            x_test = pd.DataFrame(data=x_test, index=test_index, columns=columns)
            
        if isinstance(y_train, (pd.core.series.Series)):
            y_train = y_train.to_frame()
        if isinstance(y_test, (pd.core.series.Series)):
            y_test = y_test.to_frame()
        return x_train, x_test, y_train, y_test
    

    def run(self):
        case = {"train": self.train, "test": self.test, "inference": self.inference}
        case[self.args['phase']]()

        
    def valid(self, x_valid, y_valid, n_features):
        print(f"y_valid columns: {len(y_valid.columns)}, y_valid rows: {len(y_valid.index)}")
        
        y_pred = self.model.predict(x_valid)
        if self.args.get('target_is_single') and y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        print('\n valid scores:')
        self.metric(y_pred, y_valid, n_features)
        self.export_result(x_valid, y_valid, y_pred)
        

    def test(self):
        print("testing.......... \n")
        print(f"\nmodel: {self.args.get('model')} \n")
        x_eval, y_eval = self.get_dataset(train=False)
        if isinstance(y_eval, (pd.core.series.Series)):
            y_eval = y_eval.to_frame()
        self.load_model()
        n_features = self.model.n_features_in_
        y_pred = self.model.predict(x_eval)
        if self.args.get('target_is_single') and y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        
        self.feature_importances()

        test_scores = self.metric(y_pred, y_eval, n_features)
        self.export_metric(test_scores)
        
        return y_pred
    

    # dummy part
    def inference(self):
        self.load_model()
        y_test = self.model.predict(self.data)
        return y_test
    
    
    def feature_importances(self):
        # get feature_importances or coefs
        feature_importances = getattr(self.model, self.args.get('feature_importances'))
        feature_importances = np.round(feature_importances, 3)
        if feature_importances.ndim > 1:
            feature_importances = np.squeeze(feature_importances)
        
        if self.args.get('model') == 'xgboost.XGBRegressor':
            xgb_booster = self.model.get_booster()
            feature_names = getattr(xgb_booster, self.args.get('feature_names'))
        else:
            feature_names = getattr(self.model, self.args.get('feature_names'))
            
        feature_dict = dict(map(lambda i, j : (i, j), feature_names, feature_importances))
        sorted_items = sorted(feature_dict.items(), key=lambda x:x[1], reverse=True)
        sorted_dict = dict(sorted_items)
        sorted_df = pd.DataFrame.from_dict(data=sorted_dict, orient='index', columns=[self.args.get('feature_importances')])
        print(sorted_df)
        feature_importance_path = f"./result/{self.args.get('feature_importances')}_{self.args.get('model')}_target-{self.args.get('target_name')}.xlsx"
        sorted_df.to_excel(feature_importance_path)
    

    def metric(self, prediction, valid, n_features):
        if self.args.get('model_type') == 'regressor':
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
            
            error_range = self.args.get('error_range')
            len_forecast = len(valid.columns)
            num_data = len(valid)
            r2s = []
            adj_r2s = []
            MAE = []
            MAPE = []
            MSE = []
            underestimated = []
            overestimated = []
            max_E = []
            for i in range(len_forecast):
                if isinstance(valid, (np.ndarray)):
                    valid_i = valid[:, i]
                elif isinstance(valid, (pd.core.frame.DataFrame)):
                    valid_i = valid.iloc[:, i]
                if isinstance(prediction, (np.ndarray)):
                    prediction_i = prediction[:, i]
                elif isinstance(prediction, (pd.core.frame.DataFrame)):
                    prediction_i = prediction.iloc[:, i]
                
                r2s.append(r2_score(valid_i, prediction_i))
                adj_r2s.append(1 - (1 - r2s[i]) * (num_data - 1)/(num_data - (n_features + i) - 1))
                MAE.append(mean_absolute_error(valid_i, prediction_i))
                MSE.append(mean_squared_error(valid_i, prediction_i))
                MAPE.append(mean_absolute_percentage_error(valid_i, prediction_i))
                temp_under = (1 - error_range)*valid_i > prediction_i
                underestimated.append(temp_under.values.sum()/valid_i.size)
                temp_over = (1 + error_range)*valid_i < prediction_i
                overestimated.append(temp_over.values.sum()/valid_i.size)
                max_E.append(max_error(valid_i, prediction_i))
                
            self.MAE_record.append(MAE)
            self.MSE_record.append(MSE)
            self.RMSE_record.append(np.sqrt(MSE))
            self.underestimated_record.append(underestimated)
            self.overestimated_record.append(overestimated)
            self.max_E_record.append(max_E)
            
            scores_dict = dict(
                R2 = np.round(r2s, 3),
                Adj_R2 = np.round(adj_r2s, 3),
                MAE = np.round(MAE, 3),
                MAPE = np.round(MAPE, 3),
                MSE = np.round(MSE, 3),
                RMSE = np.round(np.sqrt(MSE), 3),
                underestimated = np.round(underestimated, 3),
                overestimated = np.round(overestimated, 3),
                max_error = np.round(max_E, 3),
            )
            scores_df = pd.DataFrame.from_dict(scores_dict, orient='columns')
            print(scores_df)
            return scores_df
            
        # dummy part
        elif self.args.get('model_type') == 'classifier':
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve

            print(confusion_matrix(valid, prediction))
            print(classification_report(valid, prediction))
            print(roc_auc_score(valid, prediction))
            plot_roc_curve()

#         if 'xgb' in self.args.get('model').lower():
#             from xgboost import plot_importance;
#             plot_importance(self.model)
            
#         pyplot.show()

            
    def export_result(self, x_valid, y_valid, y_pred):
        input_concat = pd.concat([x_valid, y_valid], axis=1)
        input_concat = input_concat.reset_index()
        
        print(y_valid.shape, y_pred.shape)
        
        input_pred = pd.DataFrame(y_pred, columns=list(y_valid.columns))
        out_concat = pd.concat([input_concat, input_pred], axis=1)
        result_path = './result/'
        result_file = f"result_{self.args.get('model')}_({self.args.get('data_loader')})_target-{self.args.get('target_name')}.xlsx"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        out_concat.to_excel(os.path.join(result_path, result_file))
    
    def export_metric(self, result):
        result_path = './result/'
        result_file = f"{self.args.get('phase')}_metric_result_{self.args.get('model')}_({self.args.get('data_loader')})_target-{self.args.get('target_name')}_cv{self.args.get('cv_fold')}.xlsx"
        result.to_excel(os.path.join(result_path, result_file))
        
    def save_testset(self, x_test, y_test):
        testset = pd.concat([x_test, y_test], axis=1)
        if self.args.get('target_name') == 'dose':
            temp1 = self.args["testset_dose"]
        elif self.args.get('target_name') == 'fk':
            temp1 = self.args["testset_fk"]
            
        if self.args.get('target_is_single'):
            temp2 = f"_target-{self.args.get('end_day')}_{self.given_day}days_given.xlsx"
        elif self.args.get('target_is_next'):
            temp2 = f"_target-{self.args.get('given_day')+1}to{self.args.get('end_day')}.xlsx"
        
        test_path = temp1 + temp2
        
        if os.path.isfile(test_path):
            print(f"test data already exist: {test_path}")
        else:
            testset.to_excel(test_path)
            print(f"test data is created: {test_path}")

        
    def train(self):
        # feature selection
        cv = self.args.get('cv_fold')
        param_searching_method = self.args.get('param_searching_method')
        seed = self.args.get('seed')
        print("training.......... \n")
        
        if param_searching_method:
            
            print("parameter searching part \n")
            print("parameter searching usually consumes a lot of computing resource, so this code performs only 1 case searching unlike cv or non-cv training. \n")
            self.data = self.get_dataset()
            self.model = self.get_model()
            print(f"\nmodel: {self.args.get('model')} \n")
            
            if self.args.get('target_is_single'):
                print(f"\ntarget day: {self.args.get('end_day')}")
            else:
                print(f"\ntarget days: {self.args.get('given_day') + 1} to {self.args.get('end_day')}")
                from sklearn.multioutput import RegressorChain
                self.model = RegressorChain(self.model)
            
            x_trainval, self.x_test, y_trainval, self.y_test = self.get_train_loader(self.data)
            self.save_testset(self.x_test, self.y_test)
            
            search_params = self.args.get('search_params')
            param_searching_method_candidate = {'random': RandomizedSearchCV, 'grid': GridSearchCV}
            if param_searching_method == 'random':
                model_opt = param_searching_method_candidate[param_searching_method](self.model(), search_params, n_jobs=self.args.get('n_jobs'), cv=cv, verbose=10, random_state=seed, scoring='neg_root_mean_squared_error')
            elif param_searching_method == 'grid':
                model_opt = param_searching_method_candidate[param_searching_method](self.model(), search_params, n_jobs=self.args.get('n_jobs'), cv=cv, verbose=10, scoring='neg_root_mean_squared_error')
                
            print(f"{param_searching_method} searching, {cv}-fold cv \n")
            model_opt.fit(x_trainval, y_trainval)
            print(f"Optimized hyper parameters: {model_opt.best_params_}\n")
            print(f"Best cross validation score: {model_opt.best_score_}\n")
            
            temp = model_opt.best_params_
            temp['best_score_'] = model_opt.best_score_
            temp['scoring'] = 'neg_root_mean_squared_error'
            np.save(f"./param_search/{param_searching_method}search_{self.args.get('model')}_target-{self.args.get('target_name')}.npy", temp)
        
        elif cv:
            print("validation method: cv \n")
            
            if self.args.get('target_is_single') and not self.args.get('target_is_next'):
                period = self.args.get('end_day') - self.args.get('given_day')
            else:
                period = 1
            
            for _ in range(period):
                
                self.data = self.get_dataset()
                self.model = self.get_model()(**self.args["model_params"])
                print(f"\nmodel: {self.args.get('model')} \n")
                
                if (self.args.get('target_is_single') or self.args.get('target_is_next')) and not (self.args.get('target_is_single') and self.args.get('target_is_next')):
                    if self.args.get('target_is_single'):
                        print(f"\ntarget day: {self.given_day + 1}")
                    else:
                        print(f"\ntarget days: {self.args.get('given_day') + 1} to {self.args.get('end_day')}")
                        from sklearn.multioutput import RegressorChain
                        self.model = RegressorChain(self.model)
                elif self.args.get('target_is_single') and self.args.get('target_is_next'):
                    print(f"\ntarget day: {self.args.get('end_day')}")
                
                x_trainval, self.x_test, y_trainval, self.y_test = self.get_train_loader(self.data)
                self.save_testset(self.x_test, self.y_test)
                
                scores = cross_validate(self.model, x_trainval, y_trainval, 
                                        scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'max_error'], 
                                        cv=cv, verbose=10, return_train_score=False, )
                df_scores = np.round(pd.DataFrame(scores), 3)
                pd.set_option('display.max_columns', None);print(df_scores);pd.reset_option('display.max_columns');
                self.export_metric(df_scores)

                self.given_day += 1
        
        else:
            print("validation method: non-cv \n")
            
            if self.args.get('target_is_single') and not self.args.get('target_is_next'):
                period = self.args.get('end_day') - self.args.get('given_day')
            else:
                period = 1
            
            for _ in range(period):
                
                self.data = self.get_dataset()
                self.model = self.get_model()(**self.args["model_params"])
                print(f"\nmodel: {self.args.get('model')} \n")
                
                if (self.args.get('target_is_single') or self.args.get('target_is_next')) and not (self.args.get('target_is_single') and self.args.get('target_is_next')):
                    if self.args.get('target_is_single'):
                        print(f"\ntarget day: {self.given_day + 1}")
                    else:
                        print(f"\ntarget days: {self.args.get('given_day') + 1} to {self.args.get('end_day')}")
                        from sklearn.multioutput import RegressorChain
                        self.model = RegressorChain(self.model)
                elif self.args.get('target_is_single') and self.args.get('target_is_next'):
                    print(f"\ntarget day: {self.args.get('end_day')}")
                
                x_trainval, self.x_test, y_trainval, self.y_test = self.get_train_loader(self.data)
                self.save_testset(self.x_test, self.y_test)
                if self.args.get('validation'):
                    x_train, x_valid, y_train, y_valid = self.get_train_loader((x_trainval, y_trainval), use_scaler=False)
                else:
                    print('no validation')
                    x_train, y_train = x_trainval, y_trainval
                
                if self.args.get('training_callback') and self.args.get('validation'):
                    print(f"use callback: {self.args.get('callback_kwargs')} \n")
                    callback = get_class(self.args['training_callback'])(rounds=self.args.get('callback_kwargs')['rounds'], save_best=self.args.get('callback_kwargs')['save_best'])
                    self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[callback])
                else:
                    self.model.fit(x_train, y_train)
                    
                if self.args.get('validation'):
                    n_features = self.model.n_features_in_
                    self.valid(x_valid, y_valid, n_features)
                    
                if self.args.get('checkpoint'):
                    self.save_model()
                
                self.given_day += 1
                
                
            if self.args.get('validation') and period > 1:
                linegraph_step(
                    self.args.get('model'),
                    self.args.get('target_name'), 
                    self.args.get('given_day'), 
                    self.args.get('end_day'), 
                    self.args.get('target_is_next'), 
                    self.MAE_record, 
                    self.MSE_record, 
                    self.RMSE_record,
                    self.underestimated_record,
                    self.overestimated_record,
                    self.max_E_record,
                )
