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


def plot_dframe(dframe, title_cond, iv1, iv2, measure, title, savepath, forprinting=False, ylimit=18):

    plot_data = dframe.copy()  # preserve original

    vals = [title_cond[0], iv1[0], iv2[0]]
    slice = (title_cond[1], iv1[1], iv2[1])  # which conds to plot
    label = (title_cond[2], iv1[2], iv2[2])

    # create constant indices for consistent use in function
    tit_idx = 0
    iv1_idx = 1
    iv2_idx = 2

    # index vals to be plotted... important to keep all vals of iv determines color wheel
    plot_vals = [title_cond[tit_idx], vals[iv1_idx][slice[iv1_idx]], vals[iv2_idx][slice[iv2_idx]]]

    sns.set_theme(context='paper', style="whitegrid", font_scale=1.2,
                  rc={"axes.labelsize": 19})
    labelsize = 12

    if forprinting:
        color_palette = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]
    else:
        color_palette = sns.color_palette('colorblind', len(vals[iv1_idx]))
        color_palette = color_palette[slice[iv1_idx]]
    color_order = vals[iv1_idx]
    color_order = color_order[slice[iv1_idx]]

    # creates dataframe made up of only values for plotting
    plot_data = pd.concat(plot_data[(plot_data[label[tit_idx]] == x)] for x in plot_vals[tit_idx])
    plot_data = pd.concat(plot_data[(plot_data[label[iv1_idx]] == x)] for x in plot_vals[iv1_idx])
    plot_data = pd.concat(plot_data[(plot_data[label[iv2_idx]] == x)] for x in plot_vals[iv2_idx])

    # for idx, i_title_cond in enumerate(plot_data[label[tit_idx]].unique()):
    #     i_dframe = plot_data[(plot_data[label[tit_idx]] == i_title_cond)].copy()  # plot dframe for all title conds
    #     i_bar = sns.catplot(x=label[iv2_idx], y=measure,
    #                         hue=label[iv1_idx], hue_order=color_order, kind='bar', legend=False,
    #                         errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
    #                         data=i_dframe, ci=95, n_boot=2000, palette=color_palette,
    #                         col='decoder')
    #     i_bar.set(title=i_title_cond.upper(), xlabel=label[iv2_idx], ylabel=measure)
    #     # i_bar.ax.set_ylim(bottom=0, top=4.2)
    #     # set legend ppts: note that bbox_to_anchor is used in conjunction with loc(default='best')
    #     i_bar.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.75), facecolor=i_bar.ax.get_facecolor(),
    #                     edgecolor='1', labelspacing=.65)
    #     i_bar.tight_layout()
    #     i_bar.fig.set(dpi=400, size_inches=(10, 5))
    #     i_bar.savefig(os.path.join(savepath, f"{title}_{i_title_cond}.png"))
    #     plt.close()
    # i_dframe = plot_data[(plot_data[label[tit_idx]] == i_title_cond)].copy()  # plot dframe for all title conds

    # plots multiple plots on facegrid for each col (title_cond)
    i_bar = sns.catplot(x=label[iv2_idx], y=measure,
                        hue=label[iv1_idx], hue_order=color_order, kind='bar', legend=False,
                        errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8], sharey=True,
                        data=plot_data, ci=95, n_boot=2000, palette=color_palette,
                        col=label[tit_idx])
    # i_bar.set(xlabel=label[iv2_idx], ylim=(0, 18))
    i_bar.set(xlabel=label[iv2_idx], ylim=(0, ylimit))
    i_bar.set_titles(col_template="{col_name}")
    i_bar.set_ylabels(measure, size=labelsize)
    i_bar.set_xlabels(label[iv2_idx], size=labelsize)
    # i_bar.ax.set_ylim(bottom=0, top=4.2)
    # set legend ppts: note that bbox_to_anchor is used in conjunction with loc(default='best')
    # i_bar.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.75), facecolor=i_bar.ax.get_facecolor(),
    #                 edgecolor='1', labelspacing=.65)
    i_bar.tight_layout()
    i_bar.fig.subplots_adjust(wspace=0.09)
    i_bar.fig.set(dpi=500, size_inches=(10, 5))
    i_bar.add_legend(fontsize=labelsize)
    # i_bar.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.75), facecolor=i_bar.ax.get_facecolor(),
    #                     edgecolor='1', labelspacing=.65)
    # i_bar.savefig(os.path.join(savepath, f"{'TEST'}.png"))
    i_bar.savefig(os.path.join(savepath, f"{title}_lin.png"))
    i_bar.set(yscale='log')
    i_bar.savefig(os.path.join(savepath, f"{title}_log.png"))
    plt.close()


my_ylim = 2

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

ivs = dict(decoder=['WTA', 'PV', 'ML'],
           ori=['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique'],
           contrast=['2.5', '5.0', '10.0', '20.0', '40.0'])

for i_file in os.listdir():
    if i_file.split('.')[-1] == 'pkl':
        i_condcombo = i_file.split('.')[0]
        i_graph_path = os.path.join(graph_path, i_condcombo)
        if not os.path.exists(i_graph_path):
            os.makedirs(i_graph_path)
        with open(i_file, "rb") as input_file:
            i_pkl = pickle.load(input_file)
        i_info = i_pkl['information']
        i_data = i_pkl['all_data']
        i_data = add_cardinal_and_oblique(i_data)  # append cardinal and oblique collapsed conditions
        i_data = change_dframe_labels(i_data, 'ori', [-45, 0, 45, 90],  # replace old labels with new ones
                                      ['minus45', 'vertical', 'plus45', 'horizontal'])
        i_data = change_dframe_col_name(i_data, 'iv', 'contrast')  # more appropriate col name
        i_data = change_dframe_labels(i_data, 'contrast', list(i_data['contrast'].unique()),
                                      i_data['contrast'].unique().astype(str))
        i_data = custom_sort_dframe(i_data, ivs)  # custom sort for plotting with seaborn

        plot_dframe(i_data, title_cond=[ivs['decoder'], slice(0, len(ivs['decoder'])), 'decoder'],
                    iv1=[ivs['ori'], slice(4, 6), 'ori'],
                    iv2=[ivs['contrast'], slice(0, len(ivs['contrast'])), 'contrast'], measure='threshold',
                    savepath=i_graph_path,
                    title=f"{i_condcombo}_{my_ylim}ylim",
                    forprinting=True,
                    ylimit=my_ylim)  # forprinting sets to monochrome

print('debug')
