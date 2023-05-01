import pandas as pd
import os

# def ID_maker(value):
#     return "ID_"+str(int(value))

def get_data(file_path: str, inputs: list, given_day: int, end_day: int, target_is_single: bool, target_is_next: bool, target_name: str):
    data = pd.read_excel(file_path, index_col='등록번호')
    # NaN 제거
    data.dropna(inplace=True)
#     data = data.astype('float')

    # 필요한 열 선택
    if 'ordinal' in file_path.lower():
        dose_start = 19
        fk_start = 31
    elif 'onehot' in file_path.lower():
        dose_start = 30
        fk_start = 42      
    x_others = data.loc[:, inputs]


    if target_name == 'dose':
        if (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            x_dose = data.iloc[:, dose_start:dose_start + given_day]
            x_fk = data.iloc[:, fk_start:fk_start + given_day]
            x = pd.concat([x_others, x_fk, x_dose], axis=1)
            if target_is_single:
                y = data.iloc[:, dose_start + end_day-1]
            else:
                y = data.iloc[:, dose_start + given_day: dose_start + end_day]
        elif target_is_single and target_is_next:
            x = x_others
            y = data.iloc[:, dose_start + end_day-1]
            
    elif target_name == 'fk':
        if (target_is_single or target_is_next) and not (target_is_single and target_is_next):
            x_dose = data.iloc[:, dose_start:dose_start + given_day]
            x_fk = data.iloc[:, fk_start:fk_start + given_day]
            x = pd.concat([x_others, x_dose, x_fk], axis=1)
            if target_is_single:
                y = data.iloc[:, fk_start + end_day-1]
            else:
                y = data.iloc[:, fk_start + given_day: fk_start + end_day]
        elif target_is_single and target_is_next:
            x = x_others
            y = data.iloc[:, fk_start + end_day-1]
            

    print(f"\n input columns: {list(x.columns)}")
    print(f"\n target name: {target_name}")
#     print(f"\n inputs: \n{x}")
#     print(f"\n targets: \n{y}")
    
    pd.set_option('display.max_columns', None);
    print(f"\n inputs: \n{x.sample(1)}")
    print(f"\n targets: \n{y.sample(1)}")
    pd.reset_option('display.max_columns');
        
    return x, y


def get_test_data(file_path: str, target_name: str, target_is_single: bool, given_day: bool):
    data = pd.read_excel(file_path, index_col='등록번호')
    # NaN 제거
    data.dropna(inplace=True)

    if target_is_single:
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        x = data.iloc[:, :-given_day]
        y = data.iloc[:, -given_day:]

    print(f"\n input columns: {list(x.columns)}")
    print(f"\n target name: {target_name}")
#     print(f"\n inputs: \n{x}")
#     print(f"\n targets: \n{y}")
    
    pd.set_option('display.max_columns', None);
    print(f"\n inputs: \n{x.sample(1)}")
    print(f"\n targets: \n{y.sample(1)}")
    pd.reset_option('display.max_columns');
        
    return x, y

