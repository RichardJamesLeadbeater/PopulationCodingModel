import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import dill as pickle


def change_dframe_labels(dframe, col, old_labels=None, new_labels=None):
    # runs through old labels and replaces with new labels
    label_list = dframe[col].to_list()  # change label names in list external to dframe
    for idx, i_label in enumerate(label_list):
        for jdx in range(len(old_labels)):
            if i_label == old_labels[jdx]:  # if matches old label
                label_list[idx] = new_labels[jdx]  # replace with matching new label

    output_dframe = dframe.copy()  # preserve original if desired
    og_columns = output_dframe.columns.to_list()  # save original column order for later
    temp_columns = [x for x in og_columns]  # creates unlinked copy of list
    temp_columns.remove(col)  # remove column you wish to replace
    output_dframe = output_dframe[temp_columns]
    output_dframe[col] = label_list  # replace column with new labels
    output_dframe = output_dframe[og_columns]  # re-orders columns to original order

    return output_dframe


def change_dframe_col_name(dframe, old_name, new_name):
    col_copy = dframe[old_name].to_list()
    output_dframe = dframe.copy()
    output_dframe[new_name] = col_copy
    columns = [x if x != old_name else new_name for x in dframe.columns.to_list()]  # replaces old col with new
    return output_dframe[columns]


def add_cardinal_and_oblique(dframe):
    ori_ = 'ori'
    cardinals = dframe[(dframe[ori_] == 0) | (dframe[ori_] == 90)].copy()
    cardinals[ori_] = 'cardinal'
    obliques = dframe[(dframe[ori_] == -45) | (dframe[ori_] == 45)].copy()
    obliques[ori_] = 'oblique'
    output_dframe = pd.concat([dframe, cardinals, obliques])
    return output_dframe


def custom_sort_dframe(dframe, custom_order):
    # pull out custom orders into dicty
    # create new dicty with each label assigned a number (used to define sort order)
    custom_sort = {}
    cols = []
    for i_categ in custom_order:
        cols.append(i_categ)
        for idx, cond in enumerate(custom_order[i_categ]):
            custom_sort[cond] = idx
    output_dframe = dframe.copy()  # preserve original dframe
    output_dframe = output_dframe.sort_values(by=cols, key=lambda x: x.map(custom_sort))  # custom sort
    return output_dframe


def plot_graph_save(dframe, x, y, hue, hue_order, palette, img_name, plotname):
    bar = sns.catplot(x=x, y=y, hue=hue, hue_order=hue_order, kind='bar', legend=False,
                      errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
                      data=dframe, ci=95, n_boot=2000, palette=palette)
    bar.set(xlabel=iv, ylabel=measure)
    bar.ax.set(title=plotname.upper())
    # if any(i == name for i in ['rjl']):
    #     i_bar.ax.set_ylim(bottom=0, top=0.05)
    # set legend ppts: note that bbox_to_anchor is used in conjunction with loc(default='best')
    bar.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.75), facecolor=bar.ax.get_facecolor(),
                  edgecolor='1', labelspacing=.65)
    bar.tight_layout()
    bar.fig.set(dpi=400, size_inches=(10, 5))
    bar.savefig(os.path.join(graph_path, f"{img_name}_{exp_name}_{plotname}.png"))
    plt.close()


og_path = os.getcwd()
data_path = os.path.join(og_path, 'data')
graph_path = os.path.join(og_path, 'graphs')
for i_path in [data_path, graph_path]:
    if not os.path.exists(i_path):
        os.makedirs(i_path)

filenames = []
os.chdir(data_path)  # change directory to open files
infosets = []
datasets = []

dframe_orders = dict(decoder=['WTA', 'PV', 'ML'],
                     ori=['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique'],
                     contrast=['2.5', '5.0', '10.0', '20.0', '40.0'])

for i_file in os.listdir():
    if i_file.split('.')[-1] == 'pkl':
        i_condcombo = i_file.split('.')[0]
        i_path = os.path.join(graph_path, i_condcombo)
        if not os.path.exists(i_path):
            os.makedirs(i_path)
        with open(i_file, "rb") as input_file:
            i_pkl = pickle.load(input_file)
        i_info = i_pkl['information']
        i_data = i_pkl['all_data']
        i_data = add_cardinal_and_oblique(i_data)  # append cardinal and oblique collapsed conditions
        i_data = change_dframe_labels(i_data, 'ori', [-45, 0, 45, 90],  # replace old labels with new ones
                                      ['minus45', 'vertical', 'plus45', 'horizontal'])
        i_data = change_dframe_col_name(i_data, 'iv', 'contrast')  # more appropriate col name
        i_data = custom_sort_dframe(i_data, dframe_orders)  # custom sort for plotting with seaborn
        print('ho')
os.chdir(og_path)  # back to original path


#  density = [1, 1/1.5, 1/2, 1/2.5, 1/3]
#  fwhm = [50, 45, 40, 35, 30]
#  bounds = [[45, 45], [30, 60], [20, 70]]
#  conds - all(j == i for i in [vertical, horizontal] for j in D]
density = []

sns.set_theme()
sns.set_context('paper')
color_palette = sns.color_palette('colorblind', 6)
color_order = ['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique']
plot_data = dataset.copy()

if which_graph == 'raw':
    orientation = 'orientation'
    measure = 'threshold'
    z = 'participant'
else:
    orientation = 'ori'
    measure = 'mean'
    z = iv

all_oris = (dataset,
            'all', color_order, color_palette)
std_oris = (dataset[(dataset[orientation] != 'cardinal') & (dataset[orientation] != 'oblique')],
            'oris', color_order[0:4], color_palette[0:4])
cvso = (dataset[(dataset[orientation] == 'cardinal') | (dataset[orientation] == 'oblique')],
        'cvso', color_order[4:6], color_palette[4:6])
hvo = (dataset[(dataset[orientation] != 'cardinal') | (dataset[orientation] == 'oblique')],
       'hvo', color_order[0:2] + color_order[5:6], color_palette[0:2] + color_palette[5:6])
plot_info = [all_oris, std_oris, cvso, hvo]

for i_info in plot_info:
    if which_graph == 'summary':
        plot_graph_save(i_info[0], x=iv, y=measure, hue=orientation, hue_order=i_info[2],
                        palette=i_info[3], img_name=i_info[1], plotname='all_observers')
    elif which_graph == 'raw':
        for name in plot_data[z].unique():
            i_data = i_info[0]
            i_data = i_data[(i_data[z] == name)]
            i_bar = sns.catplot(x=iv, y=measure,
                                hue=orientation, hue_order=i_info[2], kind='bar', legend=False,
                                errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
                                data=i_data, ci=95, n_boot=2000, palette=i_info[3])
            i_bar.set(xlabel=iv, ylabel=measure)
            i_bar.ax.set(title=name.upper())
            # i_bar.ax.set_ylim(bottom=0, top=4.2)
            # set legend ppts: note that bbox_to_anchor is used in conjunction with loc(default='best')
            i_bar.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.75), facecolor=i_bar.ax.get_facecolor(),
                            edgecolor='1', labelspacing=.65)
            i_bar.tight_layout()
            i_bar.fig.set(dpi=400, size_inches=(10, 5))
            i_bar.savefig(os.path.join(graph_path, f"{i_info[1]}_{exp_name}_{name}.png"))
            plt.close()

print('debug')
