import numpy as np

rng = 30
n_guess = []
for e in range(10000):
    num = np.random.randint(1, rng + 1)
    n = 1
    guess = np.random.randint(1, rng + 1)
    while abs(guess - num) > 3:
        n += 1
        guess = np.random.randint(1, rng + 1)
    n_guess.append(n)

print(np.mean(n_guess))




