import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("/home/khatiwada/fuzzy_fd/stats/autojoin_grid_search_result.csv")

# Select a model (Change this to any model you want to filter)
model_name = "mistral"  # Change this to the desired model name
df = df[df["model_name"] == model_name]

# Plot the graph
plt.figure(figsize=(8, 5))

# Define styles for each metric
styles = {
    "avg_pr": {"label": "Precision", "color": "red", "linestyle": "-", "marker": "o"},
    "avg_recall": {"label": "Recall", "color": "blue", "linestyle": "-", "marker": "s"},
    "avg_f1": {"label": "F1-Score", "color": "green", "linestyle": "-", "marker": "d"},
}

# Plot each line with its respective style
for col, style in styles.items():
    plt.plot(df["threshold"], df[col], label=style["label"], color=style["color"],
             linestyle=style["linestyle"], marker=style["marker"])

# Labels and title
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.ylim(0, 1)  # Set Y-axis between 0 and 1
plt.grid(True, linestyle="--", alpha=0.7)
plt.title(f"Performance Metrics vs. Threshold for {model_name}", fontsize=14)

# Move the legend above the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

# Show the plot
plt.show()
plt.savefig(f"threshold_analysis_{model_name}.pdf")
