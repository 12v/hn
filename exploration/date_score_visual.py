import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(r"hn_corpus\hn_corpus.csv", encoding='latin1')

timex = df['time'][::100]
scorey = df['score'][::100]
plt.figure(figsize=(10, 5))
plt.plot(timex,scorey)[::100]

plt.xlabel("Time")
plt.ylabel("Score")
plot_file = os.path.join(script_dir, "date_vs_score.png")
plt.savefig(plot_file)

plt.show()