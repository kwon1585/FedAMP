import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./results/fedamp_accuracies.pkl', 'rb') as f:
    fedamp_accuracies = pickle.load(f)

with open('./results/fedavg_accuracies.pkl', 'rb') as f:
    fedavg_accuracies = pickle.load(f)

iterations = list(range(len(fedamp_accuracies['mean'])))

fedamp_mean = np.array(fedamp_accuracies['mean'])
fedamp_min = np.array(fedamp_accuracies['min'])
fedamp_max = np.array(fedamp_accuracies['max'])

fedavg_mean = np.array(fedavg_accuracies['mean'])
fedavg_min = np.array(fedavg_accuracies['min'])
fedavg_max = np.array(fedavg_accuracies['max'])

plt.fill_between(iterations, fedamp_min, fedamp_max, color='blue', alpha=0.2)
plt.plot(iterations, fedamp_mean, color='blue', label='FedAMP (Mean Accuracy)', linewidth=2)

plt.fill_between(iterations, fedavg_min, fedavg_max, color='red', alpha=0.2)
plt.plot(iterations, fedavg_mean, color='red', label='FedAvg (Mean Accuracy)', linewidth=2)

plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('FedAMP vs FedAvg: Accuracy over Iterations')
plt.grid(True)
plt.legend(loc='best')

plt.savefig('./results/fedamp_vs_fedavg_accuracy.png', dpi=200)
print("done.")
plt.show()
