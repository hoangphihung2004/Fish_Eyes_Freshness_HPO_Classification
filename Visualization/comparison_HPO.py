import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


asha_rd = pd.read_csv(r"")
asha_tpe = pd.read_csv(r"")

hyperband_rd = pd.read_csv(r"")
hyperband_tpe = pd.read_csv(r"")

rd = pd.read_csv(r"")
tpe = pd.read_csv(r"")



def plot_method(df, label):
    x = df["number"] + 1
    y = df["best_val_accuracy"]

    # sort theo trial
    sorted_idx = x.argsort()
    x = x.iloc[sorted_idx]
    y = y.iloc[sorted_idx]

    plt.plot(x, y, marker='o', linewidth=1.5, label=label)


plt.figure(figsize=(14, 8))

plot_method(hyperband_tpe, "TPE + Hyperband")
plot_method(hyperband_rd, "Random Search + Hyperband")
plot_method(asha_tpe, "TPE + ASHA")
plot_method(asha_rd, "Random Search + ASHA")
plot_method(tpe, "TPE")
plot_method(rd, "Random Search")


plt.xlabel("Trial Number", fontsize=13, fontweight='bold')
plt.ylabel("Validation Accuracy", fontsize=13, fontweight='bold')

max_trial = max(
    hyperband_rd["number"].max(),
    hyperband_tpe["number"].max(),
    asha_rd["number"].max(),
    asha_tpe["number"].max(),
    rd["number"].max(),
    tpe["number"].max()
)

plt.xticks(np.arange(1, max_trial + 2, 1))

plt.legend(title="Optimization Methods",
           fontsize=12,
           title_fontsize=13,
           framealpha=0.3,
           loc='lower left',
           bbox_to_anchor=(0.23, 0))


plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("", dpi=300, bbox_inches='tight')

plt.show()