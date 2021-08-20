import pandas as pd
import os
import scipy.stats as st
import mytools


def raw_to_summary(input_dir, splitter, out_dir, out_label):
    # splitter is 1x2 tuple where [0] is split and [1] is the final index after split
    os.chdir(input_dir)
    for i_file in os.listdir():
        if i_file.split(splitter[0])[-1] == splitter[1]:
            i_pickle = pd.read_pickle(i_file)
        else:
            continue
        i_raw_data = i_pickle['all_data']
        i_condition = i_file.split('.')[0]
        i_raw_data['condition'] = [i_condition] * len(i_raw_data)

        i_summary_data = {decoder: [], ori: [], iv: [], dv: []
                          }
        for j_decoder in i_raw_data[decoder].unique():
            for j_ori in i_raw_data[ori].unique():
                for j_iv in i_raw_data[iv].unique():
                    j_data = i_raw_data[(i_raw_data[decoder] == j_decoder) & (i_raw_data[ori] == j_ori)
                                        & (i_raw_data[iv] == j_iv)]
                    i_summary_data[decoder].append(j_decoder)
                    i_summary_data[ori].append(j_ori)
                    i_summary_data[iv].append(j_iv)
                    i_summary_data[dv].append(j_data[dv].mean())

        i_summary_data = pd.DataFrame.from_dict(i_summary_data)
        i_summary_data['condition'] = [i_condition] * len(i_summary_data)
        to_csv_pkl(i_summary_data, out_dir, f"{i_condition}_summary", rnd=4)


def to_csv_pkl(dframe, folder, title, rnd=2, pkl=True, csv=False):
    if csv:
        dframe.round(rnd).to_csv(os.path.join(folder, f"{title}.csv"))
    if pkl:
        dframe.to_pickle(os.path.join(folder, f"{title}.pkl"))


decoder = 'decoder'
ori = 'ori_std'
iv = 'contrast'
dv = 'threshold'

og_path = os.getcwd()
in_path = os.path.join(og_path, 'data')
raw_path = os.path.join(og_path, 'raw')
summary_path = os.path.join(og_path, 'summary')

for i_path in [raw_path, summary_path]:
    if not os.path.exists(i_path):
        os.makedirs(i_path)

raw_to_summary(in_path, ('_', 'mocs.pkl'), out_dir=summary_path, out_label='mocs')
raw_to_summary(in_path, ('_', 'staircase.pkl'), out_dir=summary_path, out_label='staircase')
z = 1
