import numpy as np
from IPython import embed
from eval_metrics import evaluate
from eval_metrics import evaluate,plot_roc,plot_acc

targets=np.random.randint(1,10,(100))
distances=np.random.randn(100)
tpr, fpr, acc, val, val_std, far = evaluate(distances,targets)
embed()
plot_roc(fpr, tpr, figure_name = './test.png')