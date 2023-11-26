import numpy as np
import matplotlib.pyplot as plt


def unsupervised_hebbian_learning(input_signal, weights, learning_rate=0.1):
    output_signal = np.dot(input_signal, weights)
    delta_weights = learning_rate * np.outer(input_signal, output_signal)
    new_weights = weights + delta_weights
    return new_weights, output_signal


def plot_weights(weights, title):
    plt.figure(figsize=(6, 4))
    plt.matshow(weights, fignum=1, cmap='viridis')
    plt.colorbar()
    plt.title(title, pad=10)
    plt.xlabel('Выходные нейроны')
    plt.ylabel('Входные нейроны')
    plt.show()



initial_weights_unsupervised = np.random.rand(3, 2) - 0.5
pattern_A = np.array([1, -1, 1])
pattern_B = np.array([-1, 1, -1])
eta = 0.3
iterations = 10
weights = initial_weights_unsupervised.copy()

for j in range(iterations):
    weights, _ = unsupervised_hebbian_learning(pattern_A, weights, learning_rate=eta)
    weights, _ = unsupervised_hebbian_learning(pattern_B, weights, learning_rate=eta)


plot_weights(initial_weights_unsupervised, "Начальные весы")
plot_weights(weights, "Финальные весы")
