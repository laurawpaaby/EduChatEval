#### -------- Purpose: -------- ####
# Visualize predicted dialogue categories from LLM simulations or classifiers.
# Supports summary tables, turn-based trend plots, bar plots, and conditional history plots.

#### -------- Inputs: -------- ####
# - DataFrame with predicted label columns
# - Optional list of columns to compare (e.g., student vs tutor)
# - Plot config options like palette, percent/count toggle, titles

#### -------- Outputs: -------- ####
# - Plots using seaborn/matplotlib
# - Summary table (as a DataFrame)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#### LINE PLOT FOR PREDICTED CATEGORIES ####

def plot_predicted_categories(df, label_columns, use_percent=True, palette="icefire", title="Predicted Category Distribution"):
    label_map = {
        'predicted_labels_student_msg': 'Student',
        'predicted_labels_tutor_msg': 'Tutor'
    }

    long_dfs = []
    for col in label_columns:
        temp = df[['turn', col]].copy()
        temp['source'] = label_map.get(col, col)
        temp.rename(columns={col: 'predicted_label'}, inplace=True)
        long_dfs.append(temp)

    long_df = pd.concat(long_dfs, ignore_index=True)

    all_labels = sorted(long_df['predicted_label'].dropna().unique())
    long_df['predicted_label'] = pd.Categorical(long_df['predicted_label'], categories=all_labels, ordered=True)

    count_df = long_df.groupby(['turn', 'source', 'predicted_label'], observed=True).size().reset_index(name='count')

    if use_percent:
        total_per_group = count_df.groupby(['turn', 'source'], observed=True)['count'].transform('sum')
        count_df['value'] = (count_df['count'] / total_per_group) * 100
        y_label = "Occurrences (%)"
        fmt = lambda y, _: f'{y:.0f}%'
        y_max = 100
    else:
        count_df['value'] = count_df['count']
        y_label = "Number of Occurrences"
        fmt = lambda y, _: f'{int(y)}'
        y_max = count_df['value'].max() + 3

    sns.set_style("whitegrid")
    g = sns.relplot(
        data=count_df,
        x='turn',
        y='value',
        hue='predicted_label',
        kind='line',
        col='source' if len(label_columns) > 1 else None,
        facet_kws={'sharey': True, 'sharex': True},
        height=4.5,
        aspect=1.5,
        marker='o',
        palette=palette,
        hue_order=all_labels
    )

    if len(label_columns) > 1:
        g.set_titles("{col_name} Messages")
    g.set_axis_labels("Turn", y_label)

    g.fig.subplots_adjust(right=0.85)
    g._legend.set_bbox_to_anchor((1.12, 0.5))
    g._legend.set_frame_on(True)
    g._legend.set_title("Predicted Category")

    for ax in g.axes.flat:
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt))

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()


#### BAR PLOT FOR PREDICTED CATEGORIES ####

def plot_category_bars(df, label_columns, use_percent=True, palette="icefire", title="Predicted Category Distribution"):
    label_map = {
        'predicted_labels_student_msg': 'Student',
        'predicted_labels_tutor_msg': 'Tutor'
    }

    long_dfs = []
    for col in label_columns:
        temp = df[[col]].copy()
        temp['source'] = label_map.get(col, col)
        temp.rename(columns={col: 'predicted_label'}, inplace=True)
        long_dfs.append(temp)

    long_df = pd.concat(long_dfs, ignore_index=True)

    all_labels = sorted(long_df['predicted_label'].dropna().unique())
    long_df['predicted_label'] = pd.Categorical(long_df['predicted_label'], categories=all_labels, ordered=True)

    count_df = long_df.groupby(['source', 'predicted_label'], observed=True).size().reset_index(name='count')

    if use_percent:
        total_per_source = count_df.groupby('source', observed=True)['count'].transform('sum')
        count_df['value'] = (count_df['count'] / total_per_source) * 100
        y_label = "Occurrences (%)"
        fmt = lambda val: f"{val:.0f}%"
    else:
        count_df['value'] = count_df['count']
        y_label = "Number of Occurrences"
        fmt = lambda val: f"{int(val)}"

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=count_df,
        x='predicted_label',
        y='value',
        hue='source',
        palette=palette,
        order=all_labels
    )

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if use_percent:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    fmt(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

    plt.legend(title="Agent")
    plt.tight_layout()
    plt.show()


#### TABLE OF SIMPLE SUMMARY STATISTICS ####

def create_prediction_summary_table(df, label_columns):
    assert 1 <= len(label_columns) <= 2, "You must provide one or two label columns."

    label_map = {
        'predicted_labels_student_msg': 'Student',
        'predicted_labels_tutor_msg': 'Tutor'
    }

    result_dfs = []
    all_categories = set()

    for col in label_columns:
        label = label_map.get(col, col.capitalize())
        value_counts = df[col].value_counts(dropna=False)
        total = value_counts.sum()
        counts = value_counts.rename(f"{label} (n)")
        percents = ((value_counts / total) * 100).round(1).astype(str) + '%'
        percents.name = f"{label} (%)"
        merged = pd.concat([counts, percents], axis=1)
        result_dfs.append(merged)
        all_categories.update(merged.index)

    full_index = pd.Index(sorted(all_categories), name="Predicted Category")
    summary_df = pd.DataFrame(index=full_index)

    for df_part in result_dfs:
        summary_df = summary_df.join(df_part, how='left')

    for col in summary_df.columns:
        if "(n)" in col:
            summary_df[col] = summary_df[col].fillna(0).astype(int)
        elif "(%)" in col:
            summary_df[col] = summary_df[col].fillna("0.0%")

    summary_df = summary_df.reset_index()
    return summary_df


#### HISTORY INTERACTION PLOT ####

def plot_previous_turn_distribution(df, focus_agent='student', use_percent=True):
    if focus_agent == 'student':
        focus_col = 'predicted_labels_student_msg'
        opposite_col = 'predicted_labels_tutor_msg'
        focus_label = 'Student'
        opposite_label = 'Tutor'
    else:
        focus_col = 'predicted_labels_tutor_msg'
        opposite_col = 'predicted_labels_student_msg'
        focus_label = 'Tutor'
        opposite_label = 'Student'

    df_sorted = df.sort_values(by=['student_id', 'turn']).copy()
    df_sorted['prev_opposite_label'] = df_sorted.groupby('student_id')[opposite_col].shift(1)
    df_filtered = df_sorted.dropna(subset=[focus_col, 'prev_opposite_label'])

    grouped = df_filtered.groupby([focus_col, 'prev_opposite_label'], observed=True).size().reset_index(name='count')

    if use_percent:
        total_per_focus = grouped.groupby(focus_col, observed=True)['count'].transform('sum')
        grouped['percentage'] = (grouped['count'] / total_per_focus) * 100
        y_col = 'percentage'
        y_label = f"Category in Previous Turn for {opposite_label} (%)"
        fmt = lambda val: f"{val:.0f}%"
    else:
        grouped['percentage'] = grouped['count']
        y_col = 'count'
        y_label = f"Category in Previous Turn for {opposite_label} (n)"
        fmt = lambda val: f"{int(val)}"

    focus_vals = sorted(df_filtered[focus_col].dropna().unique())
    prev_vals = sorted(df_filtered['prev_opposite_label'].dropna().unique())
    full_grid = pd.MultiIndex.from_product([focus_vals, prev_vals], names=[focus_col, 'prev_opposite_label']).to_frame(index=False)
    grouped = full_grid.merge(grouped, on=[focus_col, 'prev_opposite_label'], how='left').fillna(0)
    grouped['count'] = grouped['count'].astype(int)
    if use_percent:
        grouped['percentage'] = grouped.groupby(focus_col)['count'].transform(lambda x: x / x.sum() * 100).fillna(0)

    grouped = grouped.sort_values(by=[focus_col, 'prev_opposite_label'])

    sns.set_style("whitegrid")
    g = sns.catplot(
        data=grouped,
        x=focus_col,
        y=y_col,
        hue='prev_opposite_label',
        kind='bar',
        palette='icefire',
        height=6,
        aspect=2.5,
        dodge=True,
        order=focus_vals,
        hue_order=prev_vals
    )

    for patch in g.ax.patches:
        patch.set_width(patch.get_width() * 0.9)

    g.set_axis_labels(
        f"Category in Current Turn for {focus_label}",
        y_label
    )
    g.fig.suptitle(
        f"Interaction History: {focus_label} Focus",
        fontsize=15,
        fontweight='bold',
        y=0.99
    )

    if use_percent:
        g.ax.set_ylim(0, 100)
        g.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    dodge_width = 0.8 / len(prev_vals)
    for i, row in grouped.iterrows():
        x_pos = focus_vals.index(row[focus_col])
        hue_idx = prev_vals.index(row['prev_opposite_label'])
        xpos_shifted = x_pos - 0.4 + dodge_width / 2 + hue_idx * dodge_width
        height = row[y_col]
        g.ax.annotate(
            fmt(height),
            xy=(xpos_shifted, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

    g.fig.subplots_adjust(right=0.85)
    g._legend.set_bbox_to_anchor((1.12, 0.5))
    g._legend.set_frame_on(True)
    g._legend.set_title(f"{opposite_label} Category (Turn - 1)")

    plt.tight_layout()
    plt.show()
