# Visualizer
This class can be used to generate four different visualizations to analysis the interactions by. 
Every visualization besides the [Interaction Distribution plot](#interaction-distribution) can be created for either the student, the tutor or both. The [Interaction Distribution plot](#interaction-distribution) needs both student and tutor data to visualize interactions. 

#### 1. Barchart of Predicted Classes
::: educhateval.descriptive_results.display_results.plot_category_bars

Example: 
``` python
viz.plot_category_bars(
    df=annotated_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",
    use_percent=True,
    title="Distribution of Predicted Classes"
)
```
Return:
![barplot](pics/simple_bar.png)

**Parameters:**

| Name        | Type   | Description                                                           | Default   |
|-------------|--------|-----------------------------------------------------------------------|-----------|
| `df`        | `df`   | Dataframe containing predicted labels.                                | *required* |
| `student_col`| `str`  | Name of the column with predicted labels for the **student**.           | `None`  |
| `tutor_col`| `str`  |Name of the column with predicted labels for the **tutor**.                | `None`  |
| `title`      | `str`  | Title for the plot      | *"Predicted Classes"*     |




#### 2. Summary Table 
::: educhateval.descriptive_results.display_results.create_prediction_summary_table


#### 3. Predicted Classes by Turns
::: educhateval.descriptive_results.display_results.plot_predicted_categories


#### 4. Interaction Distribution { #interaction-distribution }
::: educhateval.descriptive_results.display_results.plot_previous_turn_distribution

