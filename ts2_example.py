from synthetic_data_metrics import evaluator
from synthetic_data_metrics.datasets import load_timeseries

real, synth = load_timeseries()
ev = evaluator.TS_Evaluator(real, synth, target='ACTIVITY')
print(ev.discriminative_score())
tsne = ev.t_SNE()
tsne.show()
