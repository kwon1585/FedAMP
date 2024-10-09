import pickle
import matplotlib.pyplot as plt

with open('./results/fedamp_accuracies.pkl', 'rb') as f:
    fedamp_accuracies = pickle.load(f)

with open('./results/fedavg_accuracies.pkl', 'rb') as f:
    fedavg_accuracies = pickle.load(f)

iterations = list(range(len(fedamp_accuracies)))
print(fedamp_accuracies)
print(fedavg_accuracies)

plt.plot(iterations, fedamp_accuracies, 'blue', label='FedAMP')
plt.plot(iterations, fedavg_accuracies, 'red', label='FedAvg')

plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('FedAMP vs FedAvg Accuracy')
plt.grid()
plt.legend(loc='best')

plt.savefig('./results/fedamp_vs_fedavg_accuracy.png', dpi=200)
print("done.")
plt.show()