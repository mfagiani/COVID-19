import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COUNTRY_REGION_LIST = ["Switzerland", "Italy"]
ALIGN_ZERO = True


def load_data_from_source(source='csse'):

    if source == 'csse':
        filepath_confirmed = "../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
        filepath_deaths = "../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
        filepath_recovered = "../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

    else:
        raise NotImplementedError("No other sources available yet.")

    df_confirmed = pd.read_csv(filepath_confirmed)
    df_deaths = pd.read_csv(filepath_deaths)
    df_recovered = pd.read_csv(filepath_recovered)

    return df_confirmed, df_deaths, df_recovered


def clean_data(df, country_region_list=[], align_zero=False):

    if country_region_list:

        subset = df[df['Country/Region'].isin(country_region_list)].copy()
        subset = subset.set_index('Country/Region')

        # cols_to_drop = ['Province/State', 'Country/Region', 'Lat', 'Long']
        cols_to_drop = ['Province/State', 'Lat', 'Long']
        subset = subset.drop(columns=cols_to_drop)
        subset = subset.T
        subset.index = pd.to_datetime(subset.index)

    else:
        raise ValueError("Empty country/region list.")

    if align_zero:

        support_df = subset[subset > 0]
        support_df = support_df.dropna(axis=0, how='all')

        ref_valid_index = support_df.first_valid_index()

        columns_suffix = []
        for col in support_df.columns:

            n_shift = support_df[col].isna().sum()

            if n_shift:
                support_df[col] = np.roll(support_df[col], len(support_df) - n_shift)
                support_df.rename(columns={col: f"{col} (shifted {n_shift} days)"}, inplace=True)

        # raise NotImplementedError("Feature to be implemented.")
        subset = support_df

    return subset


def plot_multi(data, cols=None, spacing=.1, same_plot=False, **kwargs):
    """ref: https://stackoverflow.com/a/11643893/5490538"""

    if same_plot:
        ax = data.plot(**kwargs)
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=2)

    else:
        plt.figure()
        # Get default color style from pandas - can be changed to any other color list
        if cols is None: cols = data.columns
        if len(cols) == 0: return
        # colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))
        c_colors = plt.get_cmap("tab10")
        colors = c_colors.colors
        # First axis
        ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
        ax.set_ylabel(ylabel=cols[0])
        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.set_ylabel(ylabel=cols[n])

            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
            # ax_new.set_ylabel(ylabel=cols[n])

            # Proper legend position
            line, label = ax_new.get_legend_handles_labels()
            lines += line
            labels += label

        ax.legend(lines, labels, loc=2)
    # plt.show()
    # return ax


if __name__ == '__main__':

    df_confirmed, df_deaths, df_recovered = load_data_from_source()

    df_confirmed = clean_data(df_confirmed, COUNTRY_REGION_LIST, align_zero=ALIGN_ZERO)
    df_deaths = clean_data(df_deaths, COUNTRY_REGION_LIST, align_zero=ALIGN_ZERO)
    df_recovered = clean_data(df_recovered, COUNTRY_REGION_LIST, align_zero=ALIGN_ZERO)

    # plot cumulative data
    plot_multi(df_confirmed,  figsize=(20, 10), title="# of confirmed", same_plot=ALIGN_ZERO)
    plot_multi(df_deaths, figsize=(20, 10), title="# of deaths", same_plot=ALIGN_ZERO)
    plot_multi(df_recovered, figsize=(20, 10), title="# of recovered", same_plot=ALIGN_ZERO)

    # plot diff from previous day
    plot_multi(df_confirmed.diff(),  figsize=(20, 10), title="Daily confirmed", same_plot=ALIGN_ZERO)
    plot_multi(df_deaths.diff(), figsize=(20, 10), title="Daily deaths", same_plot=ALIGN_ZERO)
    plot_multi(df_recovered.diff(), figsize=(20, 10), title="Daily recovered", same_plot=ALIGN_ZERO)

    # plot percentage changes from previous day
    plot_multi(df_confirmed.pct_change(),  figsize=(20, 10), title="Daily confirmed % change", same_plot=ALIGN_ZERO)
    plot_multi(df_deaths.pct_change(), figsize=(20, 10), title="Daily deaths % change", same_plot=ALIGN_ZERO)
    plot_multi(df_recovered.pct_change(), figsize=(20, 10), title="Daily recovered % change", same_plot=ALIGN_ZERO)

    # plot percentage changes from cumsum
    plot_multi(df_confirmed / df_confirmed.cumsum(),  figsize=(20, 10), title="Daily confirmed % over cumsum", same_plot=ALIGN_ZERO)
    plot_multi(df_deaths / df_deaths.cumsum(), figsize=(20, 10), title="Daily deaths % over cumsum", same_plot=ALIGN_ZERO)
    plot_multi(df_recovered / df_recovered.cumsum(), figsize=(20, 10), title="Daily recovered % over cumsum", same_plot=ALIGN_ZERO)

    plt.show()
