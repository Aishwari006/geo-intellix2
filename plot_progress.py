import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE DATA ---
# Categories (Metrics)
metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']

# The Scores (3 Sets)
# Set 1: Baseline (Zero-Shot)
run1 = [0.1744, 0.0275, 0.0161, 0.0119]
# Set 2: Improved (Few-Shot / Intermediate)
run2 = [0.2232, 0.0831, 0.0642, 0.0520]
# Set 3: Final (Refined / Fine-Tuned)
run3 = [0.2471, 0.1017, 0.0374, 0.0223]

# --- 2. SETUP THE BARS ---
x = np.arange(len(metrics))  # Label locations
width = 0.25  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the 3 groups side-by-side
rects1 = ax.bar(x - width, run1, width, label='Baseline (Zero-Shot)', color='#ff9999')
rects2 = ax.bar(x,        run2, width, label='Improved (Few-Shot)',   color='#66b3ff')
rects3 = ax.bar(x + width, run3, width, label='Final (Refined)',      color='#99ff99')

# --- 3. STYLING ---
ax.set_ylabel('BLEU Score')
ax.set_title('GeoIntellix Performance Progress')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# --- 4. SAVE ---
plt.tight_layout()
plt.savefig('progress_graph.png', dpi=300)
print("Graph saved as 'progress_graph.png'")