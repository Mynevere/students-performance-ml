import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Adjust the path if your dataset is in another folder
df = pd.read_csv("StudentsPerformance.csv")

# Calculate average scores by gender
avg_scores = df.groupby("gender")[["math score", "reading score", "writing score"]].mean().reset_index()

# Reshape for visualization
avg_scores_melted = avg_scores.melt(id_vars="gender", var_name="Subject", value_name="Average Score")

plt.figure(figsize=(8,6))
sns.barplot(
    data=avg_scores_melted,
    x="Subject",
    y="Average Score",
    hue="gender",
    palette=["#ff8ba7", "#7da0fa"]
)

plt.title("Performance Comparison by Gender (Female vs Male)", fontsize=13, weight="bold")
plt.ylabel("Average Score")
plt.xlabel("Subject")
plt.ylim(0, 100)
plt.legend(title="Gender", labels=["Female", "Male"])
plt.grid(axis="y", linestyle="--", alpha=0.6)

output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gender_comparison.png"))
plt.show()

print("Figure 'gender_comparison.png' successfully created in outputs/figures/")
