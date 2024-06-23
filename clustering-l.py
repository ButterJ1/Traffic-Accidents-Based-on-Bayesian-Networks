import numpy as np
import pandas as pd
import glob
import os

path = 'D:/東海大學/112學年/下學期/5572_風險分析與管理/101~112年臺北市A1及A2類交通事故明細'

all_files = glob.glob(os.path.join(path, '*.csv'))

specific_culm_head = [
    '發生年', '發生月', '發生日', '發生時', '發生分', '處理別-編號', '區序', '發生地點', '死亡人數(24小時內)', '死亡人數(2-30日內)',
    '受傷人數', '當事人序號', '當事者區分(類別)', '天候', '道路照明設備', '道路類別', '速限-速度限制', '道路型態', '事故位置', '路面鋪裝',
    '路面狀態', '路面缺陷', '障礙物', '視距', '號誌種類', '號誌動作', '車道劃分設施及分向設施', '快車道或一般車道間',
    '快慢車道間', '路面邊線', '事故類型及型態', '屬(性)別', '年齡', '受傷程度', '主要傷處', '保護裝備', '行動電話', '車輛用途',
    '當事者行動狀態', '駕照狀態', '駕照種類', '飲酒情形', '車輛撞擊部位1', '車輛撞擊部位2', '肇因碼-個別',
    '肇因碼-主要', '肇事逃逸', '職業', '旅次目的', '座標-X', '座標-Y'
]

dfs_by_header = {}
columns_info = {}

for filename in all_files:
    try:
        df = pd.read_csv(filename, encoding='big5hkscs', low_memory=False)

        print(f"\nNumber of columns in the DataFrame: {df.shape[0]} columns")
        
        original_columns = list(df.columns)
        
        if len(original_columns) == len(specific_culm_head):
            column_mapping = {orig_col: specific_culm_head[i] for i, orig_col in enumerate(original_columns)}
            df = df.rename(columns=column_mapping)
            df['座標-X'] = pd.to_numeric(df['座標-X'], errors='coerce')
            df['座標-Y'] = pd.to_numeric(df['座標-Y'], errors='coerce')
        else:
            if '性別' in df.columns:
                df.rename(columns={'性別': '屬(性)別'}, inplace=True)
            if '發生年度' in df.columns:
                df.rename(columns={'發生年度': '發生年'}, inplace=True)
            if '發生時-Hours' in df.columns:
                df.rename(columns={'發生時-Hours': '發生時'}, inplace=True)
            if '肇事地點' in df.columns:
                df.rename(columns={'肇事地點': '發生地點'}, inplace=True)
            if '死亡人數' in df.columns:
                df.rename(columns={'死亡人數': '死亡人數(24小時內)'}, inplace=True)
            if '車種' in df.columns:
                df.rename(columns={'車種': '當事者區分(類別)'}, inplace=True)
            print(f"Columns in file {filename} do not match expected length, keeping original column names.")

        rows_to_drop = df[df['當事者區分(類別)'] == "33"].index
        df = df.drop(rows_to_drop)

        rows_to_drop = df[df['速限-速度限制'] == 500.0].index
        df = df.drop(rows_to_drop)
        
        header_tuple = tuple(original_columns)
        
        if header_tuple not in dfs_by_header:
            dfs_by_header[header_tuple] = []
        dfs_by_header[header_tuple].append(df)
                
        columns_info[filename] = df.shape[0]
    
    except UnicodeDecodeError as e:
        print(f"Error reading {filename} with Big5 encoding: {e}")
    except Exception as e:
        print(f"An error occurred while reading {filename}: {e}")

combined_dfs = []

for header_tuple, dfs in dfs_by_header.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_dfs.append(combined_df)

if combined_dfs:
    final_combined_df = pd.concat(combined_dfs, ignore_index=True)
else:
    print("No DataFrames to concatenate.")

final_combined_df.replace(" ", np.nan, inplace=True)

exclude_columns = ['區序', '發生地點', '當事者區分(類別)', '座標-X', '座標-Y']

for col in final_combined_df.columns:
    if col not in exclude_columns:
        final_combined_df[col] = pd.to_numeric(final_combined_df[col], errors='coerce').astype('Int64')

final_combined_df['發生時間'] = pd.to_datetime(final_combined_df['發生年'] + 1911, format='%Y', errors='coerce') + \
                 pd.to_timedelta(final_combined_df['發生月'] - 1, unit='m') + \
                 pd.to_timedelta(final_combined_df['發生日'] - 1, unit='d') + \
                 pd.to_timedelta(final_combined_df['發生時'], unit='h') + \
                 pd.to_timedelta(final_combined_df['發生分'], unit='m')

def categorize_hours(hour):
    if pd.isna(hour):
        return np.nan
    if 0 <= hour <= 6:
        return '0-6'
    elif 7 <= hour <= 11:
        return '7-11'
    elif 12 <= hour <= 18:
        return '12-18'
    elif 19 <= hour <= 23:
        return '19-23'

final_combined_df['時段'] = final_combined_df['發生時'].apply(categorize_hours)

def categorize_age(age):
    if pd.isna(age):
        return '不明'
    age = int(age)
    if age < 18:
        return '18歲以下'
    elif 18 <= age <= 19:
        return '18-19歲'
    elif 20 <= age <= 24:
        return '20-24歲'
    elif 25 <= age <= 29:
        return '25-29歲'
    elif 30 <= age <= 34:
        return '30-34歲'
    elif 35 <= age <= 39:
        return '35-39歲'
    elif 40 <= age <= 44:
        return '40-44歲'
    elif 45 <= age <= 49:
        return '45-49歲'
    elif 50 <= age <= 54:
        return '50-54歲'
    elif 55 <= age <= 59:
        return '55-59歲'
    elif 60 <= age <= 64:
        return '60-64歲'
    elif 65 <= age <= 69:
        return '65-69歲'
    elif 70 <= age <= 74:
        return '70-74歲'
    elif 75 <= age <= 79:
        return '75-79歲'
    elif 80 <= age <= 84:
        return '80-84歲'
    elif 85 <= age <= 89:
        return '85-89歲'
    elif age >= 90:
        return '90歲以上'

final_combined_df['年齡層'] = final_combined_df['年齡'].apply(categorize_age)

specific_reindex = [
    '發生時間', '發生年', '發生月', '發生日', '時段', '發生時', '發生分', '處理別-編號', '區序', '發生地點', '死亡人數(24小時內)', '死亡人數(2-30日內)', '受傷人數', 
    '天候', '道路照明設備', '道路類別', '速限-速度限制', '道路型態', '事故位置', 
    '路面鋪裝', '路面狀態', '路面缺陷', '障礙物', '視距', '號誌種類', '號誌動作', '車道劃分設施及分向設施', 
    '快車道或一般車道間', '快慢車道間', '路面邊線', '事故類型及型態', 
    '屬(性)別', '年齡層', '年齡', 
    '受傷程度', '主要傷處', '保護裝備', '行動電話', '當事人序號', '當事者區分(類別)', 
    '車輛用途', '當事者行動狀態', '駕照狀態', '駕照種類', '飲酒情形', 
    '車輛撞擊部位1', '車輛撞擊部位2', '肇因碼-個別', '肇因碼-主要', '肇事逃逸', '職業', 
    '旅次目的', 
    '座標-X', '座標-Y',
]

final_df = final_combined_df.reindex(columns=specific_reindex, fill_value=np.nan)
print("final_df: ", final_df.dtypes)

print("Head of final combined DataFrame:")
print(final_df.head(0))
print(f"\nNumber of columns in the final combined DataFrame: {final_df.shape[0]} columns")

output_path = 'D:/東海大學/112學年/下學期/5572_風險分析與管理'
final_df.to_csv(os.path.join(output_path, 'combined_data.csv'), index=False, encoding='UTF-8')
print(f"Combined data saved to {os.path.join(output_path, 'combined_data.csv')}")


"""
most_frequent_items = {}

for var in specific_reindex:
    frequency = final_df[var].value_counts(dropna=False)
    max_freq = frequency.max()
    most_frequent = frequency[frequency == max_freq].index.tolist()
    
    if pd.isna(most_frequent[0]):
        second_max_freq = frequency[frequency < max_freq].max()
        most_frequent = frequency[frequency == second_max_freq].index.tolist()
    
    most_frequent_items[var] = most_frequent

for var, items in most_frequent_items.items():
    print(f"Variable: {var}")
    print(f"Most Frequent Items: {items}")
    print("\n")
"""
"""
variable_combination = ['區序', '發生地點', '肇因碼-個別', '肇因碼-主要', '年齡層', '當事人序號', '當事者區分(類別)', '事故類型及型態', '時段', '發生年']
# variable_combination = ['區序', '肇因碼-主要', '年齡層', '當事者區分(類別)', '事故類型及型態', '時段', '發生年']

combo_freq = final_df[variable_combination].value_counts(dropna=False).reset_index(name='Frequency')

most_frequent_combos = combo_freq[combo_freq['Frequency'] == combo_freq['Frequency'].max()]

for i in range(len(most_frequent_combos)):
    print(f"Most Frequent Combination {i + 1}:")
    print(most_frequent_combos.iloc[i])
    
    condition = pd.Series([True] * len(final_df))
    for col in variable_combination:
        col_condition = (final_df[col] == most_frequent_combos.iloc[i][col]) | (final_df[col].isna() & pd.isna(most_frequent_combos.iloc[i][col]))
        condition &= col_condition
    
    frequent_records = final_df[condition]
    
    print("Records with the Most Frequent Combination:")
    print(frequent_records)
    if (i != len(most_frequent_combos)-1):
        print("\n" + "="*50 + "\n")

"""