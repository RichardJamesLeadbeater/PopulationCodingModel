from scipy import stats as st
import mytools
import numpy as np
import random
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

    data_plot = dframe.copy()  # preserve original

    vals = [title_cond[0], iv1[0]]
    slicer = [title_cond[1], iv1[1]]  # which conds to plot
    label = [title_cond[2], iv1[2]]

    # create constant indices for consistent use in function
    tit_idx = 0
    iv1_idx = 1

    # index vals to be plotted... but still retain all vals of iv as it determines color wheel
    plot_vals = [title_cond[tit_idx], vals[iv1_idx][slicer[iv1_idx]]]

    if iv2:
        vals.append(iv2[0])
        slicer.append(iv2[1])
        label.append(iv2[2])
        iv2_idx = 2
        plot_vals.append(vals[iv2_idx][slicer[iv2_idx]])

    if forprinting:
        colour_palette = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]  # b&w
    else:
        colour_palette = sns.color_palette('colorblind', len(vals[iv1_idx]))
        colour_palette = colour_palette[slicer[iv1_idx]]
    colour_order = vals[iv1_idx]
    colour_order = colour_order[slicer[iv1_idx]]

    # creates dataframe made up of only values for plotting
    data_plot = pd.concat(data_plot[(data_plot[label[tit_idx]] == x)] for x in plot_vals[tit_idx])
    data_plot = pd.concat(data_plot[(data_plot[label[iv1_idx]] == x)] for x in plot_vals[iv1_idx])
    if iv2:
        data_plot = pd.concat(data_plot[(data_plot[label[iv2_idx]] == x)] for x in plot_vals[iv2_idx])

    # plots multiple plots on facegrid for each col (title_cond)
    if iv2:
        i_bar = sns.catplot(x=label[iv2_idx], y=measure,
                        hue=label[iv1_idx], hue_order=colour_order, kind='bar', legend=False,
                        errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
                        sharey=True, sharex=True,
                        data=data_plot, ci=68, palette=colour_palette,
                        col=label[tit_idx], col_wrap=3)
    else:
        i_bar = sns.catplot(x=label[iv1_idx], y=measure,
                            kind='point', legend_out=True, sharey=True,
                            data=data_plot, ci=None, color=[0.2, 0.2, 0.2, 0.5],
                            col=label[tit_idx], col_wrap=3)

    # i_bar.set(xlabel=label[iv2_idx], ylim=(0, 18))
    i_bar.set_titles(col_template="{col_name}", y=.98, size=title_size, weight='bold')
    i_bar.set_ylabels(measure, size=label_size)
    if iv2:
        i_bar.set_xlabels(label[iv2_idx], size=label_size)
    else:
        i_bar.set_xlabels(label[iv1_idx], size=label_size)
    for j in i_bar.axes.flatten():
        j.tick_params(labelleft=True, labelbottom=True)
        j.set_yticklabels(j.get_yticklabels(), size=tick_size)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    i_bar.fig.subplots_adjust(wspace=width_space, hspace=height_space)
    i_bar.add_legend(fontsize=legend_fontsize)
    i_bar.tight_layout()
    i_bar.fig.set(dpi=250)
    # i_bar.savefig(os.path.join(savepath, f"{'TEST'}.png"))
    i_bar.savefig(os.path.join(savepath, f"{title}_lin.png"))
    i_bar.set(yscale='log', ylim=ylim_log)
    for j in i_bar.axes.flatten():
            j.tick_params(labelleft=True, labelbottom=True, which='both')
            j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
            j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    i_bar.savefig(os.path.join(savepath, f"{title}_log.png"))
    plt.close()


def init_cols(cols, val=None):
    if val is None:
        val = []
    output = {}
    for i_col in cols:
        output[i_col] = val.copy()
    return output


def get_ratio(dataf, conds=('', ''), categ='ori'):
    """calculate ratio across ? for each condition"""
    x_df = dataf[dataf[categ] == conds[0]]
    x = x_df[dv].to_list()[0]
    y_df = dataf[dataf[categ] == conds[1]]
    y = y_df[dv].to_list()[0]
    ratio = x / y
    # return np.mean(ratio)
    return ratio


def get_each_ratio(dframe, conds=None, output_type=None):
    if output_type is None:
        output_type = 'table'
        cols = which_ratios
    elif output_type == 'dframe':
        cols = ['ratio', 'value']
    if not conds:
        if output_type == 'table':
            ratio = init_cols(cols)
            # ratio['v/h'].append(get_ratio(dframe, ['vertical', 'horizontal'], categ=orientation))
            # ratio['m45/p45'].append(get_ratio(dframe, ['minus45', 'plus45'], categ=orientation))
            ratio['o/c'].append(get_ratio(dframe, ['oblique', 'cardinal'], categ=orientation))
    else:
        if not isinstance(conds, list):
            conds = [conds]
        ratio = init_cols([c for c in conds] + cols)
        for i in dframe[conds[0]].unique():
            i_data = dframe[dframe[conds[0]] == i]
            if len(i_data) == 0:
                continue
            if len(conds) == 1:
                ratio[conds[0]].append(i)
                if output_type == 'table':
                    # ratio['v/h'].append(get_ratio(i_data, ['vertical', 'horizontal'], categ=orientation))
                    # ratio['m45/p45'].append(get_ratio(i_data, ['minus45', 'plus45'], categ=orientation))
                    ratio['o/c'].append(get_ratio(i_data, ['oblique', 'cardinal'], categ=orientation))
                elif output_type == 'dframe':
                    # ratio['ratio'].append('v/h')
                    # ratio['value'].append(get_ratio(i_data, ['vertical', 'horizontal'], categ=orientation))
                    # ratio['ratio'].append('m45/p45')
                    # ratio['value'].append(get_ratio(i_data, ['minus45', 'plus45'], categ=orientation))
                    ratio['ratio'].append('o/c')
                    ratio['value'].append(get_ratio(i_data, ['oblique', 'cardinal'], categ=orientation))

            elif len(list(conds)) > 1:
                for j in dframe[conds[1]].unique():
                    j_data = i_data[i_data[conds[1]] == j]
                    if len(j_data) == 0:
                        continue
                    else:
                        ratio[conds[0]].append(i)
                        ratio[conds[1]].append(j)
                        # ratio['v/h'].append(get_ratio(j_data, ['vertical', 'horizontal'], categ=orientation))
                        # ratio['m45/p45'].append(get_ratio(j_data, ['minus45', 'plus45'], categ=orientation))
                        ratio['o/c'].append(get_ratio(j_data, ['oblique', 'cardinal'], categ=orientation))

    return pd.DataFrame.from_dict(ratio)


def mean_across_cond(dframe, condition=None):
    if not condition:
        output = init_cols([orientation, dv, 'stdev', 'sem'])
    else:
        output = init_cols([orientation, condition, dv, 'stdev', 'sem'])
    for i in dframe[orientation].unique():
        ori = i
        if not condition:  # used when taking mean across all conds for each observer
            i_data = dframe[(dframe[orientation] == ori)]
            append_dicty(output, [ori, i_data[dv].mean(), i_data[dv].std(),
                                  st.sem(i_data[dv])])
        else:
            for cond in dframe[condition].unique():  # used when taking mean across all observers for each cond
                i_data = dframe[(dframe[orientation] == ori) & (dframe[condition] == cond)]
                append_dicty(output, [ori, cond, i_data[dv].mean(),
                                      i_data[dv].std(), st.sem(i_data[dv])])
    return pd.DataFrame.from_dict(output)


def append_dicty(dicty, val_list):
    # requires val_list to be identical length to dicty with each val matching each key
    for idx, key in enumerate(dicty):
        dicty[key].append(val_list[idx])


def get_overall_mean(ratio_data):
    all_conds_all_observers = {}
    for col in ['v/h', 'm45/p45', 'o/c']:
        all_conds_all_observers[col] = ratio_data[col].mean()
    return pd.DataFrame(all_conds_all_observers, index=[dv])


def to_csv_pkl(dframe, folder, title, rnd=5, pkl=True, csv=True):
    if csv:
        dframe.round(rnd).to_csv(os.path.join(folder, f"{title}.csv"))
    if pkl:
        dframe.to_pickle(os.path.join(folder, f"{title}.pkl"))


def resample(dataset, n_samples=1000):
    dataset = np.asarray(dataset)
    # pull out resampled data for n_samples
    resampled = []
    for i in range(n_samples):
        resampled.append(random.choices(dataset, k=len(dataset)))
    return np.asarray(resampled)


def bootstrap(dframe, resample_measure):
    og_data = dframe[resample_measure]
    bs_data = resample(og_data.to_list(), n_samples=1000)

    if resample_measure == 'stdev':  # correct for bias in bs stdev
        bs_data = bs_data + (og_data.std() - np.mean(bs_data))

    return bs_data


def calc_mean_of_ratios(values):
    # put ratios on log-scale to allow us to take the mean then convert back
    return 10 ** np.mean(np.log10(values))


def get_mean_of_ratios(dframe, condition, ratio_cols=None):
    if ratio_cols is None:
        ratio_cols = which_ratios
    # output = init_cols([condition, 'ratio', 'mean_ratio'])
    output_table = init_cols([condition] + ratio_cols)
    for cond in dframe[condition].unique():  # keep separate to keep col lengths equal
        output_table[condition].append(cond)
    for i_ratio in ratio_cols:
        for cond in dframe[condition].unique():
            i_data = dframe[(dframe[condition] == cond)][i_ratio].to_numpy()
            output_table[i_ratio].append(calc_mean_of_ratios(i_data))
    return pd.DataFrame(output_table)


def summary_to_dframe(dframe, change_cols, new_col):
    keep_cols = [i for i in dframe.columns if all(j != i for j in change_cols)]
    output = init_cols(keep_cols + [new_col] + ['values'])
    for i_change in change_cols:
        output[new_col] += [i_change] * len(dframe)
        output['values'] += dframe[i_change].to_list()
        for i_keep in keep_cols:
            output[i_keep] += dframe[i_keep].to_list()

    return pd.DataFrame(output)


def get_ylim_logscale(dframe, dframe_dv='mean'):

    if dframe[dframe_dv].min() > 0.1:
        minval = (dframe[dframe_dv].min() / 1.1)
        return minval, None

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


def load_seaborn_prefs(style="ticks", context="paper"):
    axes_color = [0.2, 0.2, 0.2, 0.95]
    sns.set_theme(style=style, context=context,
                  rc={'axes.edgecolor': axes_color, 'xtick.color': axes_color, 'ytick.color': axes_color,
                      'axes.linewidth': 1, 'legend.title_fontsize': 0, 'legend.fontsize': 13, 'patch.linewidth': 1.2,
                      'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1,
                      'lines.linewidth': 2.3}
                  )


load_seaborn_prefs()
legend_fontsize = 18
label_size = 18
tick_size = 15
title_size = 19
width_space = .2
height_space = .3
lin_ymin = 0.0
lin_ymax = None

which_ratios = ['o/c']

og_path = os.getcwd()
data_path = os.path.join(og_path, 'data')  # raw data
summary_path = os.path.join(og_path, 'summary')
mean_path = os.path.join(summary_path, 'mean')
graph_path = os.path.join(og_path, 'graphs')
ratio_path = os.path.join(og_path, 'ratios')
for i_path in [data_path, graph_path, ratio_path]:
    if not os.path.exists(i_path):
        os.makedirs(i_path)

filenames = []
os.chdir(data_path)  # change directory to open files
infosets = []
datasets = []

ivs = dict(decoder=['WTA', 'PV', 'ML'],
           ori=['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique'],
           contrast=['2.5', '5.0', '10.0', '20.0', '40.0'])

iv = 'contrast'
dv = 'threshold'
orientation = 'ori'
ylimit_log = (0.4, 7)

for i_file in os.listdir():
    if i_file.split('.')[-1] != 'pkl':
        continue
    else:
        i_condition = i_file.split('.')[0]
        i_graph_path = os.path.join(graph_path, i_condition)
        i_ratio_path = os.path.join(ratio_path, i_condition)
        i_mean_path = os.path.join(mean_path, i_condition)
        for i_path in [i_graph_path, i_ratio_path, i_mean_path]:
            if not os.path.exists(i_path):
                os.makedirs(i_path)
        # with open(i_file, "rb") as input_file:
        #     try:
        #         i_pkl = pickle.load(input_file)
        #     except:
        #         print(f"File failed to be unpickled:  {i_file}")
        #         continue
        #     else:
        #         pass
        # i_info = i_pkl['information']
        # i_data = i_pkl['all_data']
        # i_data = add_cardinal_and_oblique(i_data)  # append cardinal and oblique collapsed conditions
        # i_data = change_dframe_labels(i_data, 'ori', [-45, 0, 45, 90],  # replace old labels with new ones
        #                               ['minus45', 'vertical', 'plus45', 'horizontal'])
        # i_data = change_dframe_col_name(i_data, 'iv', iv)  # more appropriate col name
        # i_data = change_dframe_labels(i_data, iv, list(i_data[iv].unique()),
        #                               i_data[iv].unique().astype(str))
        # i_data = custom_sort_dframe(i_data, ivs)  # custom sort for plotting with seaborn

        i_summary = pd.read_pickle(os.path.join(summary_path, f"{i_condition}_allsummary.pkl"))
        # i_summary = change_dframe_col_name(i_summary, 'iv', iv)
        # i_summary = add_cardinal_and_oblique(i_summary)  # append cardinal and oblique collapsed conditions
        i_summary = add_cardinal_and_oblique(i_summary)
        i_summary = change_dframe_labels(i_summary, 'ori', [-45, 0, 45, 90],  # replace old labels with new ones
                                      ['minus45', 'vertical', 'plus45', 'horizontal'])
        i_summary = change_dframe_col_name(i_summary, 'iv', 'contrast')  # more appropriate col name
        i_summary = change_dframe_labels(i_summary, 'contrast', list(i_summary['contrast'].unique()),
                                         i_summary['contrast'].unique().astype(str))
        i_summary = custom_sort_dframe(i_summary, ivs)  # custom sort for plotting with seaborn

        # useful to compare ratios that change as a function of the IV, and across participants
        each_ratio = get_each_ratio(i_summary, ['decoder', 'contrast'])
        to_csv_pkl(each_ratio, i_ratio_path, f"{i_condition}_allratios", pkl=True, csv=False)

        plot_data = summary_to_dframe(each_ratio, which_ratios, 'ratio')

        i_bar = sns.catplot(x=iv, y='values',
                            hue='decoder', hue_order=plot_data['decoder'].unique().tolist(),
                            kind='point', legend=False,
                            capsize=.04, sharey=True, palette='colorblind',
                            data=plot_data, n_boot=None, col='ratio')
        # change size of markers on line plot
        all_ymin = 1000
        all_ymax = -1
        for ax in i_bar.axes:
            for subax in ax:
                i_ylim = subax.get_ylim()
                if i_ylim[0] < all_ymin:
                    all_ymin = i_ylim[0]
                if i_ylim[1] > all_ymax:
                    all_ymax = i_ylim[1]
                for i_collection in subax.collections:
                    i_collection.set_sizes(i_collection.get_sizes() / 1.8)

        i_bar.set(ylim=(0.0, 7))
        # i_bar.set(ylim=(0.0, np.ceil(all_ymax)))
        # i_bar.fig.suptitle(f"Orientation ratios", size=18, y=.98)
        i_bar.set_titles(col_template="{col_name}", size=title_size, y=1, weight='bold')
        i_bar.set_xlabels(iv, size=label_size)
        i_bar.set_ylabels('ratio', size=label_size)
        i_bar.add_legend(fontsize=legend_fontsize)
        for j in i_bar.axes.flatten():
            j.tick_params(labelleft=True, labelbottom=True)
            j.set_yticklabels(j.get_yticklabels(), size=tick_size)
            j.set_xticklabels(j.get_xticklabels(), size=tick_size)
        i_bar.fig.subplots_adjust(wspace=width_space, hspace=height_space)
        i_bar.tight_layout()
        i_bar.fig.set(dpi=250)
        i_bar.savefig(os.path.join(i_ratio_path, f"{i_condition}_decoders_ratios.png"))
        i_bar.savefig(os.path.join(i_graph_path, f"{i_condition}_decoders_ratios.png"))
        # i_bar.set(yscale='log', ylim=(all_ymin, np.ceil(all_ymax)))
        i_bar.set(yscale='log', ylim=ylimit_log)
        for j in i_bar.axes.flatten():  # removes y-axis minor ticks
            j.tick_params(labelleft=True, labelbottom=True, which='both')
            j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
            j.set_xticklabels(j.get_xticklabels(), size=tick_size)
        i_bar.savefig(os.path.join(i_ratio_path, f"{i_condition}_decoders_ratios_log.png"))
        i_bar.savefig(os.path.join(i_graph_path, f"{i_condition}_decoders_ratios_log.png"))
        plt.close()

        # useful on tasks where ratios change as a function of the IV, but not across participants
        summary_mean_iv = mean_across_cond(i_summary, iv)
        to_csv_pkl(summary_mean_iv, i_mean_path, f"{i_condition}_iv", rnd=6, pkl=False, csv=True)
        # this method gets ratio of mean thresholds
        ratios_of_means_iv = get_each_ratio(summary_mean_iv, iv)
        to_csv_pkl(ratios_of_means_iv, i_ratio_path, f"{i_condition}_iv_ratiosofmeans", pkl=False)

        # this method gets the mean of the ratios for each observer
        means_of_ratios_iv = get_mean_of_ratios(each_ratio, iv)
        to_csv_pkl(means_of_ratios_iv, i_ratio_path, f"{i_condition}_iv_meansofratios", pkl=False)

        # useful on tasks where ratios change across participants, but not as a function of the IV
        summary_mean_obs = mean_across_cond(i_summary, 'decoder')
        to_csv_pkl(summary_mean_obs, i_mean_path, f"{i_condition}_obs", rnd=6, pkl=False, csv=True)
        mean_ratio_obs = get_each_ratio(summary_mean_obs, 'decoder')
        to_csv_pkl(mean_ratio_obs, i_ratio_path, f"{i_condition}_obs_ratio", pkl=False, csv=True)

        # only useful on tasks where ratios do not change as a function of the IV
        summary_mean_all = mean_across_cond(i_summary)
        to_csv_pkl(summary_mean_all, i_mean_path, f"{i_condition}_overall", rnd=6, pkl=False, csv=True)
        mean_ratio_all = get_each_ratio(summary_mean_all)
        to_csv_pkl(mean_ratio_all, i_ratio_path, f"{i_condition}_overall_ratio", pkl=False, csv=True)


print('')
