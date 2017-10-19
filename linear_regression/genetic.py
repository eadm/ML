import numpy as np
import ml


def solve(train_p, train_c, f, step):
    pop_size = 10
    d = train_p[0].size

    population = np.random.rand(pop_size, d)

    for i in range(step):
        new_population = np.zeros((pop_size, d))
        for j in range(pop_size):
            np.random.shuffle(population)
            [i1, i2, i3] = np.random.randint(0, 9, size=3)
            a, b, c = population[i1], population[i2], population[i3]

            an = [a + f * (b - c), a]
            af = np.zeros(d)
            for k in range(d):
                af[k] = an[np.random.randint(0, 2)][k]

            if ml.mse_w(af, train_p, train_c) < ml.mse_w(a, train_p, train_c):
                new_population[j] = af
            else:
                new_population[j] = a

        population = new_population

    w = population[0]
    for i in range(1, pop_size):
        if ml.mse_w(population[i], train_p, train_c) < ml.mse_w(w, train_p, train_c):
            w = population[i]

    print ml.mse_w(w, train_p, train_c)

    return w
