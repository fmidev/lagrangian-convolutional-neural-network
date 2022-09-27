from pincast_verif.metrics.rapsd import RapsdMetric
import numpy as np
import matplotlib.pyplot as plt

lts, n_sample = [3,6,12], 10
dummy = np.random.random(size=(10,36,256,256))


rapsy = RapsdMetric(leadtimes=lts, im_size=(256,256))

for sample in range(n_sample):
    rapsy.accumulate(x_pred = dummy[sample], x_obs=dummy[sample])

val, name = rapsy.compute()

plt.plot(val)
plt.savefig("aa.png")