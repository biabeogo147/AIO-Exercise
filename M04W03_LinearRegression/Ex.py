import random
import numpy as np
from matplotlib import pyplot as plt
random.seed(0)  # please do not remove this line


def load_data_from_file(fileName="advertising.csv"):
    data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
    features_X = data[:, :3]
    features_X = np.c_[np.ones((features_X.shape[0], 1)), features_X]
    sales_Y = data[:, 3]
    return features_X, sales_Y


def create_individual(n=4, bound=10):
    individual = np.random.uniform(-bound / 2, bound / 2, n)
    return individual


def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss = np.multiply((y_hat - sales_Y), (y_hat - sales_Y)).mean()
    return loss


def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1 / (1 + loss)
    return fitness_value


def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(len(individual1)):
        if random.random() < crossover_rate:
            individual1_new[i], individual2_new[i] = individual2[i], individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.05):
    individual_m = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] += random.gauss(0, 1)
    return individual_m


def initializePopulation(m):
    population = [create_individual() for _ in range(m)]
    return population


def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if index2 != index1:
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]
    return individual_s


def create_new_population(old_population, elitism=2, gen=1):
    new_population = []
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)
    if gen % 1 == 0:
        print(" Best loss :", compute_loss(sorted_population[m - 1]), " with chromsome : ", sorted_population[m - 1])
    while len(new_population) < m - elitism:
        individual1 = selection(sorted_population, m)
        individual2 = selection(sorted_population, m)
        individual1, individual2 = crossover(individual1, individual2)
        individual1 = mutate(individual1)
        individual2 = mutate(individual2)
        new_population.append(individual1)
        new_population.append(individual2)
    for ind in sorted_population[m - elitism:]:
        new_population.append(ind)
    return new_population, compute_loss(sorted_population[m - 1])


def run_GA():
    m = 600
    losses_list = []
    n_generations = 100
    population = initializePopulation(m)
    for i in range(n_generations):
        population, best_loss = create_new_population(population, elitism=2, gen=i)
        losses_list.append(best_loss)
    visualize_predict_gt(population)
    return losses_list


def visualize_loss(losses_list):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_list, label='Loss over generations')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('Loss vs. Generations')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_predict_gt(population):
    sorted_population = sorted(population, key=compute_fitness)
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])
    estimated_prices = features_X.dot(theta)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.plot(sales_Y, c='green', label='Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    features_X, sales_Y = load_data_from_file()

    print(features_X[:5, :])
    print(sales_Y.shape)

    individual = create_individual()
    print(individual)

    individual = [4.09, 4.82, 3.10, 4.02]
    fitness_score = compute_fitness(individual)
    print(fitness_score)

    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    individual1, individual2 = crossover(individual1, individual2, 2.0)
    print(" individual1 : ", individual1)
    print(" individual2 : ", individual2)

    before_individual = [4.09, 4.82, 3.10, 4.02]
    after_individual = mutate(individual, mutation_rate=2.0)
    print(before_individual == after_individual)

    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    old_population = [individual1, individual2]
    new_population, _ = create_new_population(old_population, elitism=2, gen=1)

    losses_list = run_GA()
    visualize_loss(losses_list)
