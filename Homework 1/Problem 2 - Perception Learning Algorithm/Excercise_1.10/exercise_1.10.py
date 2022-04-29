import random
import numpy as np
import matplotlib.pyplot as plt

# 0 - tail; 1 - head
def flip_m_coin_n_time(m,n):
    ans = np.random.randint(low = 0, high = 2, size = (m,n))
    return ans
def get_fraction(m,n):
    flips = flip_m_coin_n_time(m,n)
    c1 = flips[0]
    c_rand = flips[random.randint(0, m - 1)]
    c_min = flips[np.argmin(flips.sum(axis=1))]
    # print(flips)
    # print(np.argmin(flips.sum(axis=1)))
    v1 = np.sum(c1) / n
    v_rand = np.sum(c_rand) / n
    v_min = np.sum(c_min) / n
    return v1, v_rand, v_min

def plot_hist_fraction(data, title):
    plt.hist(data, bins=10)
    plt.title(title)
    plt.xlabel("Fraction of head")
    plt.ylabel("Count")
    plt.show()

def plot_hoeffding(data, n, title):
    # n: sample size
    #0.5 is the out-of-sample error
    e_in_minus_e_out = abs(data - 0.5)
    print(e_in_minus_e_out)
    probability = []
    epsilon = np.arange(0,1,0.001)
    for e in epsilon:
        probability.append(np.sum(e_in_minus_e_out > e) / e_in_minus_e_out.shape[0])
    plt.plot(epsilon, probability)
    plt.plot(epsilon, 2 * np.exp(-2 * (epsilon**2) * 10))
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Probability")
    plt.show()
def main():
    m = 1000 # num coins
    n = 10 # each coin flip n times
    num_exp = 1000 # num experiment
    v1_all = np.ones(num_exp)
    v_rand_all = np.ones(num_exp)
    v_min_all = np.ones(num_exp)
    for i in range(num_exp):
        v1_all[i], v_rand_all[i], v_min_all[i] = get_fraction(m,n)
    #question a
    print(np.mean(v1_all), np.mean(v_rand_all), np.mean(v_min_all))
    print("Printing histogram...")
    # question b
    plot_hist_fraction(v1_all, "Histogram of V1 Distributions")
    plot_hist_fraction(v_rand_all, "Histogram of Vrand Distributions")
    plot_hist_fraction(v_min_all, "Histogram of Vmin Distributions")
    #question c
    plot_hoeffding(v1_all, n, "Hoeffding of V1")
    plot_hoeffding(v_rand_all, n, "Hoeffding of Vrand")
    plot_hoeffding(v_min_all, n, "Hoeffding of Vmin")
if __name__ == "__main__":
    main()