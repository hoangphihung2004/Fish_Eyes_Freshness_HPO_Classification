import matplotlib.pyplot as plt
import numpy as np


data = [
    {'name': 'MB-BE', 'author': 'Prasetyo et al.', 'year': '2022', 'accuracy': 63.21},
    {'name': 'VGG19 + ANN', 'author': 'Yildiz et al.', 'year': '2024', 'accuracy': 77.30},
    {'name': 'Handcrafted Features', 'author': 'Hoang et al.', 'year': '2026', 'accuracy': 77.56},
    {'name': 'Hybrid DL-ML', 'author': 'Hoang et al.', 'year': '2026', 'accuracy': 85.99},
    {'name': 'Proposed Cross-Model Fusion', 'author': 'Our study', 'year': '', 'accuracy': 89.59}
]


methods = []
for i in data:
    if i['author'] == 'Our study':
        methods.append(f"{i['name']} ({i['author']})")
    else:
        methods.append(f"{i['name']} ({i['author']}, {i['year']})")

accuracy = [i['accuracy'] for i in data]


cmap = plt.get_cmap('tab10')

bar_colors = [cmap(i % 10) for i in range(len(data))]


best_idx = next(i for i, d in enumerate(data) if d['name'] == 'Proposed Cross-Model Fusion')
bar_colors[best_idx] = '#006400'


fig, ax1 = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(methods))

bars = ax1.bar(
    x_pos,
    accuracy,
    color=bar_colors,
    width=0.35,
    edgecolor='black',
    linewidth=0.6,
    zorder=2
)

ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

min_acc = min(accuracy)
max_acc = max(accuracy)

ax1.set_ylim(min_acc - 5, max_acc + 3)
ax1.set_xticks([])


for bar in bars:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f'{h:.2f}%',
        ha='center',
        va='bottom',
        fontsize=14,
        fontweight='bold',
        bbox=dict(
            facecolor='white',
            alpha=0.65,
            edgecolor='none',
            boxstyle='round,pad=0.25'
        )
    )

ax1.grid(axis='y', linestyle=':', color='gray', zorder=0)


ax1.legend(
    bars,
    methods,
    title="Methodology",
    title_fontsize=13,
    fontsize=12,
    loc='upper left',
    framealpha=0.95
)

plt.tight_layout()
plt.savefig("", dpi=300, bbox_inches='tight')
plt.show()