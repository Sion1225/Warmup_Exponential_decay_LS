from TF_warmup_exponential import TF_warmup_exponential

import numpy as np
import matplotlib.pyplot as plt

scheduler = TF_warmup_exponential(max_lr=0.01, min_lr=0.001, num_warmup=50, a=0.1)

steps = np.arange(0, 500, 1)
learning_rates = [np.array(scheduler(step)) for step in steps]

plt.figure(figsize=(10, 6))
plt.plot(steps, learning_rates)
plt.title('Learning Rate Schedule')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()