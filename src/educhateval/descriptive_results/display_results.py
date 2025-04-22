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
def plot_predicted_categories(
    df,
    student_col=None,
    tutor_col=None,
    use_percent=True,
    palette="icefire",
    title="Predicted Category Distribution",
):
    if not student_col and not tutor_col:
        raise ValueError("You must provide at least one of student_col or tutor_col.")

    # Prepare long format
    long_dfs = []
    if student_col:
        temp = df[["turn", student_col]].copy()
        temp["source"] = "Student"
        temp.rename(columns={student_col: "predicted_label"}, inplace=True)
        long_dfs.append(temp)
    if tutor_col:
        temp = df[["turn", tutor_col]].copy()
        temp["source"] = "Tutor"
        temp.rename(columns={tutor_col: "predicted_label"}, inplace=True)
        long_dfs.append(temp)

    long_df = pd.concat(long_dfs, ignore_index=True)

    all_labels = sorted(long_df["predicted_label"].dropna().unique())
    long_df["predicted_label"] = pd.Categorical(
        long_df["predicted_label"], categories=all_labels, ordered=True
    )

    count_df = (
        long_df.groupby(["turn", "source", "predicted_label"], observed=True)
        .size()
        .reset_index(name="count")
    )

    if use_percent:
        total_per_group = count_df.groupby(["turn", "source"], observed=True)[
            "count"
        ].transform("sum")
        count_df["value"] = (count_df["count"] / total_per_group) * 100
        y_label = "Occurrences (%)"
        fmt = lambda y, _: f"{y:.0f}%"
        y_max = 100
    else:
        count_df["value"] = count_df["count"]
        y_label = "Number of Occurrences"
        fmt = lambda y, _: f"{int(y)}"
        y_max = count_df["value"].max() + 3

    sns.set_style("whitegrid")
    g = sns.relplot(
        data=count_df,
        x="turn",
        y="value",
        hue="predicted_label",
        kind="line",
        col="source" if student_col and tutor_col else None,
        facet_kws={"sharey": True, "sharex": True},
        height=4.5,
        aspect=1.5,
        marker="o",
        palette=palette,
        hue_order=all_labels,
    )

    if student_col and tutor_col:
        g.set_titles("{col_name} Messages")
    g.set_axis_labels("Turn", y_label)

    g.fig.subplots_adjust(right=0.85)
    g._legend.set_bbox_to_anchor((1.12, 0.5))
    g._legend.set_frame_on(True)
    g._legend.set_title("Predicted Category")

    for ax in g.axes.flat:
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt))

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.95)
    plt.tight_layout()
    plt.show()


#### BAR PLOT FOR PREDICTED CATEGORIES ####
def plot_category_bars(
    df,
    student_col=None,
    tutor_col=None,
    use_percent=True,
    palette="icefire",
    title="Predicted Category Distribution",
):
    if not student_col and not tutor_col:
        raise ValueError("You must provide at least one of student_col or tutor_col.")

    long_dfs = []
    if student_col:
        temp = df[[student_col]].copy()
        temp["source"] = "Student"
        temp.rename(columns={student_col: "predicted_label"}, inplace=True)
        long_dfs.append(temp)
    if tutor_col:
        temp = df[[tutor_col]].copy()
        temp["source"] = "Tutor"
        temp.rename(columns={tutor_col: "predicted_label"}, inplace=True)
        long_dfs.append(temp)

    long_df = pd.concat(long_dfs, ignore_index=True)

    all_labels = sorted(long_df["predicted_label"].dropna().unique())
    long_df["predicted_label"] = pd.Categorical(
        long_df["predicted_label"], categories=all_labels, ordered=True
    )

    count_df = (
        long_df.groupby(["source", "predicted_label"], observed=True)
        .size()
        .reset_index(name="count")
    )

    if use_percent:
        total_per_source = count_df.groupby("source", observed=True)["count"].transform(
            "sum"
        )
        count_df["value"] = (count_df["count"] / total_per_source) * 100
        y_label = "Occurrences (%)"
        fmt = lambda val: f"{val:.0f}%"
    else:
        count_df["value"] = count_df["count"]
        y_label = "Number of Occurrences"
        fmt = lambda val: f"{int(val)}"

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=count_df,
        x="predicted_label",
        y="value",
        hue="source",
        palette=palette,
        order=all_labels,
    )

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=15, fontweight="bold")

    if use_percent:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    fmt(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.legend(title="Agent")
    plt.tight_layout()
    plt.show()


#### TABLE OF SIMPLE SUMMARY STATISTICS ####
def create_prediction_summary_table(df, student_col=None, tutor_col=None):
    if not student_col and not tutor_col:
        raise ValueError("You must provide at least one of student_col or tutor_col.")

    result_dfs = []
    all_categories = set()

    if student_col:
        student_counts = df[student_col].value_counts(dropna=False)
        total = student_counts.sum()
        counts = student_counts.rename("Student (n)")
        percents = ((student_counts / total) * 100).round(1).astype(str) + "%"
        percents.name = "Student (%)"
        merged = pd.concat([counts, percents], axis=1)
        result_dfs.append(merged)
        all_categories.update(merged.index)

    if tutor_col:
        tutor_counts = df[tutor_col].value_counts(dropna=False)
        total = tutor_counts.sum()
        counts = tutor_counts.rename("Tutor (n)")
        percents = ((tutor_counts / total) * 100).round(1).astype(str) + "%"
        percents.name = "Tutor (%)"
        merged = pd.concat([counts, percents], axis=1)
        result_dfs.append(merged)
        all_categories.update(merged.index)

    full_index = pd.Index(sorted(all_categories), name="Predicted Category")
    summary_df = pd.DataFrame(index=full_index)

    for df_part in result_dfs:
        summary_df = summary_df.join(df_part, how="left")

    for col in summary_df.columns:
        if "(n)" in col:
            summary_df[col] = summary_df[col].fillna(0).astype(int)
        elif "(%)" in col:
            summary_df[col] = summary_df[col].fillna("0.0%")

    summary_df = summary_df.reset_index()
    return summary_df


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_previous_turn_distribution(
    df, student_col=None, tutor_col=None, focus_agent="student", use_percent=True
):
    """
    Plot the distribution of predicted categories in the previous turn of the *opposite* agent. Both student and tutor is required.
    """

    if focus_agent not in ["student", "tutor"]:
        raise ValueError("focus_agent must be either 'student' or 'tutor'.")

    if focus_agent == "student":
        if not student_col or not tutor_col:
            raise ValueError(
                "Both student_col and tutor_col must be provided when focus_agent='student'."
            )
        focus_col = student_col
        opposite_col = tutor_col
        focus_label = "Student"
        opposite_label = "Tutor"
    else:
        if not student_col or not tutor_col:
            raise ValueError(
                "Both student_col and tutor_col must be provided when focus_agent='tutor'."
            )
        focus_col = tutor_col
        opposite_col = student_col
        focus_label = "Tutor"
        opposite_label = "Student"

    # Prepare shifted column
    df_sorted = df.sort_values(by=["student_id", "turn"]).copy()
    df_sorted["prev_opposite_label"] = df_sorted.groupby("student_id")[
        opposite_col
    ].shift(1)
    df_filtered = df_sorted.dropna(subset=[focus_col, "prev_opposite_label"])

    # Count combinations
    grouped = (
        df_filtered.groupby([focus_col, "prev_opposite_label"], observed=True)
        .size()
        .reset_index(name="count")
    )

    if use_percent:
        total_per_focus = grouped.groupby(focus_col, observed=True)["count"].transform(
            "sum"
        )
        grouped["percentage"] = (grouped["count"] / total_per_focus) * 100
        y_col = "percentage"
        y_label = f"Category in Previous Turn for {opposite_label} (%)"
        fmt = lambda val: f"{val:.0f}%"
    else:
        grouped["percentage"] = grouped["count"]
        y_col = "count"
        y_label = f"Category in Previous Turn for {opposite_label} (n)"
        fmt = lambda val: f"{int(val)}"

    # Ensure all category combinations are represented
    focus_vals = sorted(df_filtered[focus_col].dropna().unique())
    prev_vals = sorted(df_filtered["prev_opposite_label"].dropna().unique())
    full_grid = pd.MultiIndex.from_product(
        [focus_vals, prev_vals], names=[focus_col, "prev_opposite_label"]
    ).to_frame(index=False)
    grouped = full_grid.merge(
        grouped, on=[focus_col, "prev_opposite_label"], how="left"
    ).fillna(0)
    grouped["count"] = grouped["count"].astype(int)
    if use_percent:
        grouped["percentage"] = (
            grouped.groupby(focus_col)["count"]
            .transform(lambda x: x / x.sum() * 100)
            .fillna(0)
        )

    grouped = grouped.sort_values(by=[focus_col, "prev_opposite_label"])

    # Plot
    sns.set_style("whitegrid")
    g = sns.catplot(
        data=grouped,
        x=focus_col,
        y=y_col,
        hue="prev_opposite_label",
        kind="bar",
        palette="icefire",
        height=6,
        aspect=2.5,
        dodge=True,
        order=focus_vals,
        hue_order=prev_vals,
    )

    # Adjust bar width
    for patch in g.ax.patches:
        patch.set_width(patch.get_width() * 0.9)

    # Labels and title
    g.set_axis_labels(f"Category in Current Turn for {focus_label}", y_label)
    g.fig.suptitle(
        f"Interaction History: {focus_label} Focus",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    if use_percent:
        g.ax.set_ylim(0, 100)
        g.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Annotate values (including 0s)
    dodge_width = 0.8 / len(prev_vals)
    for i, row in grouped.iterrows():
        x_pos = focus_vals.index(row[focus_col])
        hue_idx = prev_vals.index(row["prev_opposite_label"])
        xpos_shifted = x_pos - 0.4 + dodge_width / 2 + hue_idx * dodge_width
        height = row[y_col]
        g.ax.annotate(
            fmt(height),
            xy=(xpos_shifted, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    g.fig.subplots_adjust(right=0.85)
    g._legend.set_bbox_to_anchor((1.12, 0.5))
    g._legend.set_frame_on(True)
    g._legend.set_title(f"{opposite_label} Category (Turn - 1)")

    plt.tight_layout()
    plt.show()
