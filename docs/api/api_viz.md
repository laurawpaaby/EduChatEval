# Visualizer
Module for generating four different visualizations to analysis the interactions by. 
Every visualization besides the *4. Interaction Distribution* plot can be created for either the student, the tutor or both. The Interaction Distribution plot requires both student and tutor data to visualize interactions. 

***1. Barchart of Predicted Classes***
::: educhateval.descriptive_results.display_results.plot_category_bars

Example usage: 
``` python
viz.plot_category_bars(
    df=annotated_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",
    use_percent=True,
    title="Distribution of Predicted Classes"
)
```

**Parameters:**

| Name          | Type              | Description                                                                      | Default              |
|---------------|-------------------|----------------------------------------------------------------------------------|----------------------|
| `df`          | `DataFrame`       | The input DataFrame containing predicted categories for student and/or tutor.   | *required*           |
| `student_col` | `str` or `None`   | Name of the column with **student**-predicted labels. Optional.                 | `None`               |
| `tutor_col`   | `str` or `None`   | Name of the column with **tutor**-predicted labels. Optional.                   | `None`               |
| `palette`     | `str`             | Color palette used for the plot. Optional.                                       | `"icefire"`          |
| `title`       | `str`             | Title of the plot. Optional.                                                     | `"Predicted Classes"` |

**Returns:**

| Name | Type    | Description                                                               |
|------|---------|---------------------------------------------------------------------------|
| —    | `None`  | Displays the plot using `matplotlib.pyplot.show()`. No object is returned. |


***2. Summary Table*** 
::: educhateval.descriptive_results.display_results.create_prediction_summary_table

Example usage: 
``` python
summary = viz.create_summary_table(
    df=annotated_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg"
)

print(summary)
```

**Parameters:**

| Name          | Type              | Description                                                                      | Default    |
|---------------|-------------------|----------------------------------------------------------------------------------|------------|
| `df`          | `DataFrame`       | The input DataFrame containing predicted categories for student and/or tutor.   | *required* |
| `student_col` | `str` or `None`   | Name of the column with **student**-predicted labels. Optional.                 | `None`     |
| `tutor_col`   | `str` or `None`   | Name of the column with **tutor**-predicted labels. Optional.                   | `None`     |

**Returns:**

| Name         | Type        | Description                                                                           |
|--------------|-------------|---------------------------------------------------------------------------------------|
| `summary_df` | `DataFrame` | A summary table with counts and percentages for each predicted category. Splits by student and tutor (if provided). Missing values are filled with 0. |


***3. Predicted Classes by Turns***
::: educhateval.descriptive_results.display_results.plot_predicted_categories

Example usage: 
```python
plot_predicted_categories(
        df=annotated_df,
        student_col="predicted_labels_student_msg",
        tutor_col="predicted_labels_tutor_msg",
        title="Predicted Category Distribution"
    )
```

**Parameters:**

| Name          | Type              | Description                                                                 | Default              |
|---------------|-------------------|-----------------------------------------------------------------------------|----------------------|
| `df`          | `DataFrame`       | Input DataFrame with turn-level predicted labels for student and/or tutor. | *required*           |
| `student_col` | `str` or `None`   | Name of the column containing **student**-predicted categories. Optional.  | `None`               |
| `tutor_col`   | `str` or `None`   | Name of the column containing **tutor**-predicted categories. Optional.    | `None`               |
| `use_percent` | `bool`            | Whether to plot percentage values (`True`) or raw counts (`False`).        | `True`               |
| `palette`     | `str`             | Color palette used for the plot. Optional.                                  | `"icefire"`          |
| `title`       | `str`             | Title of the plot. Optional.                                                | `"Predicted Classes"` |

**Returns:**

| Name | Type    | Description                                                               |
|------|---------|---------------------------------------------------------------------------|
| —    | `None`  | Displays the plot using `matplotlib.pyplot.show()`. No object is returned. |


***4. Interaction Distribution***
::: educhateval.descriptive_results.display_results.plot_previous_turn_distribution

Example usage:
```python
viz.plot_history_interaction(
    df=annotated_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",
    focus_agent="tutor",
    use_percent=True
)
```

**Parameters:**

| Name          | Type              | Description                                                                                      | Default       |
|---------------|-------------------|--------------------------------------------------------------------------------------------------|---------------|
| `df`          | `DataFrame`       | Input DataFrame including turn-level predicted labels for both student and tutor.              | *required*    |
| `student_col` | `str` or `None`   | Column name containing **student**-predicted categories.                                        | *required*    |
| `tutor_col`   | `str` or `None`   | Column name containing **tutor**-predicted categories.                                          | *required*    |
| `focus_agent` | `str`             | Determines whether to analyze the student or tutor perspective. Options: "student" or "tutor". | `"student"`   |
| `use_percent` | `bool`            | If `True`, the y-axis will display percentages; otherwise raw counts are shown.                | `True`        |
| `palette`     | `str`             | Color palette used for the plot. Optional.                                                      | `"icefire"`    |

**Returns:**

| Name | Type    | Description                                                               |
|------|---------|---------------------------------------------------------------------------|
| —    | `None`  | Displays the plot using `matplotlib.pyplot.show()`. No object is returned. |