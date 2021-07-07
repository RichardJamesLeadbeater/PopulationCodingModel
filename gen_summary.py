import pandas as pd
import os
import scipy.stats as st
import mytools


def to_csv_pkl(dframe, folder, title, rnd=2):
    dframe.round(rnd).to_csv(os.path.join(folder, f"{title}.csv"))
    dframe.to_pickle(os.path.join(folder, f"{title}.pkl"))


og_path = os.getcwd()
in_path = os.path.join(og_path, 'data')
raw_path = os.path.join(og_path, 'raw')
summary_path = os.path.join(og_path, 'summary')

for i_path in [raw_path, summary_path]:
    if not os.path.exists(i_path):
        os.makedirs(i_path)

# join all rawdata .pkl files into dframe
os.chdir(in_path)
raw_data = []
for i_file in os.listdir():
    if i_file.split('.')[-1] == 'pkl':
        i_pickle = pd.read_pickle(i_file)
    else:
        continue
    i_raw_data = i_pickle['all_data']
    i_condition = i_file.split('.')[0]
    i_raw_data['condition'] = [i_condition] * len(i_raw_data)

    i_summary_data = {'decoder': [], 'ori': [], 'iv': [], 'threshold': [],
                      }
    for j_decoder in i_raw_data['decoder'].unique():
        for j_ori in i_raw_data['ori'].unique():
            for j_iv in i_raw_data['iv'].unique():
                j_data = i_raw_data[(i_raw_data['decoder'] == j_decoder) & (i_raw_data['ori'] == j_ori)
                                    & (i_raw_data['iv'] == j_iv)]
                i_summary_data['decoder'].append(j_decoder)
                i_summary_data['ori'].append(j_ori)
                i_summary_data['iv'].append(j_iv)
                i_summary_data['threshold'].append(j_data['threshold'].mean())

    i_summary_data = pd.DataFrame.from_dict(i_summary_data)
    i_summary_data['condition'] = [i_condition] * len(i_summary_data)

    to_csv_pkl(i_summary_data, summary_path, f"{i_condition}_allsummary", rnd=4)
