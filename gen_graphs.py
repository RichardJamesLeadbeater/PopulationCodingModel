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


def load_seaborn_prefs(style="ticks", context="talk"):
    axes_color = [0.2, 0.2, 0.2, 0.95]
    sns.set_theme(style=style, context=context,
                  rc={'axes.edgecolor': axes_color, 'xtick.color': axes_color, 'ytick.color': axes_color,
                      'axes.linewidth': 1, 'legend.title_fontsize': 0, 'legend.fontsize': 13, 'patch.linewidth': 1.2,
                      'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1}
                  )


def get_ylim_logscale(dframe, dframe_dv='mean'):
    # auto log scale for facetgrid can cut off smallest values, this puts ylim to suitable range
    logscale_minval = dframe[dframe_dv].min() * (10 ** dframe[dframe_dv].min())
    # leave default to auto_val (None)
    if 1 > logscale_minval >= 0.1:
        log_ymin = 0.1  # if in range of 0.1 to 1 then set min to 0.1
    elif 0.1 > logscale_minval >= 0.01:
        log_ymin = 0.01
    elif 0.01 > logscale_minval >= 0.001:
        log_ymin = 0.001
    else:
        log_ymin = None
    return log_ymin, None


def plot_dframe(dframe, measure, title_cond, iv1, iv2=None, title=None, savepath=None, forprinting=False,
                ylim_lin=18, ylim_log=0.01):

    plot_data = dframe.copy()  # preserve original

    vals = [title_cond[0], iv1[0]]
    slice = [title_cond[1], iv1[1]]  # which conds to plot
    label = [title_cond[2], iv1[2]]

    # create constant indices for consistent use in function
    tit_idx = 0
    iv1_idx = 1

    # index vals to be plotted... but still retain all vals of iv as it determines color wheel
    plot_vals = [title_cond[tit_idx], vals[iv1_idx][slice[iv1_idx]]]

    if iv2:
        vals.append(iv2[0])
        slice.append(iv2[1])
        label.append(iv2[2])
        iv2_idx = 2
        plot_vals.append(vals[iv2_idx][slice[iv2_idx]])

    if forprinting:
        color_palette = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]  # b&w
    else:
        color_palette = sns.color_palette('colorblind', len(vals[iv1_idx]))
        color_palette = color_palette[slice[iv1_idx]]
    color_order = vals[iv1_idx]
    color_order = color_order[slice[iv1_idx]]

    # creates dataframe made up of only values for plotting
    plot_data = pd.concat(plot_data[(plot_data[label[tit_idx]] == x)] for x in plot_vals[tit_idx])
    plot_data = pd.concat(plot_data[(plot_data[label[iv1_idx]] == x)] for x in plot_vals[iv1_idx])
    if iv2:
        plot_data = pd.concat(plot_data[(plot_data[label[iv2_idx]] == x)] for x in plot_vals[iv2_idx])

    load_seaborn_prefs()
    # plots multiple plots on facegrid for each col (title_cond)
    if iv2:
        i_bar = sns.catplot(x=label[iv2_idx], y=measure,
                        hue=label[iv1_idx], hue_order=color_order, kind='bar', legend_out=True,
                        errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8], sharey=True,
                        data=plot_data, ci='sd', palette=color_palette,
                        col=label[tit_idx], col_wrap=3)
    else:
        i_bar = sns.catplot(x=label[iv1_idx], y=measure,
                            kind='point', legend_out=True, sharey=True,
                            data=plot_data, ci=None, color=[0.2, 0.2, 0.2, 0.5],
                            col=label[tit_idx], col_wrap=3)

    # i_bar.set(xlabel=label[iv2_idx], ylim=(0, 18))
    i_bar.set_titles(col_template="{col_name}", y=.98, size=16, weight='bold')
    i_bar.set_ylabels(measure)
    if iv2:
        i_bar.set_xlabels(label[iv2_idx])
    else:
        i_bar.set_xlabels(label[iv1_idx])
    for j in i_bar.axes.flatten():
        j.tick_params(labelleft=True)
        j.set_yticklabels(j.get_yticklabels(), size=13)
        j.set_xticklabels(j.get_xticklabels(), size=13)
    i_bar.fig.subplots_adjust(wspace=0.18, hspace=0.20)
    i_bar.tight_layout()
    i_bar.fig.set(dpi=250)
    # i_bar.savefig(os.path.join(savepath, f"{'TEST'}.png"))
    i_bar.savefig(os.path.join(savepath, f"{title}_lin.png"))
    i_bar.set(yscale='log', ylim=ylim_log)
    i_bar.savefig(os.path.join(savepath, f"{title}_log.png"))
    plt.close()


# START
my_ylim = None

og_path = os.getcwd()
summary_path = os.path.join(og_path, 'summary')
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
    if i_file.split('.')[-1] != 'pkl':
        continue
    else:
        i_condition = i_file.split('.')[0]
        i_graph_path = os.path.join(graph_path, i_condition)
        if not os.path.exists(i_graph_path):
            os.makedirs(i_graph_path)
        with open(i_file, "rb") as input_file:
            try:
                i_pkl = pickle.load(input_file)
            except:
                print(f"File failed to be unpickled:  {i_file}")
                continue
            else:
                pass
        i_info = i_pkl['information']
        i_data = i_pkl['all_data']
        i_data = add_cardinal_and_oblique(i_data)  # append cardinal and oblique collapsed conditions
        i_data = change_dframe_labels(i_data, 'ori', [-45, 0, 45, 90],  # replace old labels with new ones
                                      ['minus45', 'vertical', 'plus45', 'horizontal'])
        i_data = change_dframe_col_name(i_data, 'iv', 'contrast')  # more appropriate col name
        i_data = change_dframe_labels(i_data, 'contrast', list(i_data['contrast'].unique()),
                                      i_data['contrast'].unique().astype(str))
        i_data = custom_sort_dframe(i_data, ivs)  # custom sort for plotting with seaborn

        i_summary = pd.read_pickle(os.path.join(summary_path, f"{i_condition}_allsummary.pkl"))

        plot_dframe(i_data, title_cond=[ivs['decoder'], slice(0, len(ivs['decoder'])), 'decoder'],
                    iv1=[ivs['ori'], slice(4, 6), 'ori'],
                    iv2=[ivs['contrast'], slice(0, len(ivs['contrast'])), 'contrast'], measure='threshold',
                    savepath=i_graph_path,
                    title=f"{i_condition}_{my_ylim}ylim",
                    forprinting=False,
                    ylim_log=get_ylim_logscale(i_summary, 'threshold'))  # forprinting sets to monochrome

        # create oblique index graphs
        # create oblique index graphs
        plot_data_oi = {'decoder': [], 'contrast': [], 'oblique index': [], 'ori': []}
        for i_decoder in i_data['decoder'].unique():
            for i_con in i_data['contrast'].unique():
                j_data = i_data[(i_data['decoder'] == i_decoder) & (i_data['contrast'] == i_con)]
                cardinal = j_data[(j_data['ori'] == 'cardinal')]['threshold'].mean()
                oblique = j_data[(j_data['ori'] == 'oblique')]['threshold'].mean()

                plot_data_oi['decoder'].append(i_decoder)
                plot_data_oi['contrast'].append(i_con)
                plot_data_oi['oblique index'].append(oblique / cardinal)
                plot_data_oi['ori'].append(None)  # not used but need col for custom sort
        plot_data_oi = pd.DataFrame(plot_data_oi)
        plot_data_oi = custom_sort_dframe(plot_data_oi, ivs)

        plot_dframe(plot_data_oi, title_cond=[ivs['decoder'], slice(0, len(ivs['decoder'])), 'decoder'],
                    iv1=[ivs['contrast'], slice(0, len(ivs['contrast'])), 'contrast'],
                    measure='oblique index',
                    savepath=i_graph_path,
                    title=f"{i_condition}_OI",
                    forprinting=False,
                    ylim_log=get_ylim_logscale(plot_data_oi, 'oblique index'))
