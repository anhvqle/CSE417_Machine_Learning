import numpy as np

def random_pop(n):
    return np.random.uniform(-1,1,n)

f_x = lambda x: x ** 2
print(random_pop(10))

def get_a_and_b(x1, x2):
    a = x1 + x2
    b = x1 * x2
    return a, b


# Calculate a, b
a = b = 0
itr = 50000 # generate D
for i in range(itr):
    #generate random D
    temp = random_pop(2)
    x1, x2 = temp[0], temp[1]
    a_d, b_d = get_a_and_b(x1, x2)
    # g = a_d * x + b_d
    a += a_d
    b += b_d
a /= itr
b /= itr

print('d is',a,b)


num_pop = 100000
pop = random_pop(num_pop)

# Calculate Eout
e_out = 0
for i in range(itr):
    #generate random D
    temp = random_pop(2)
    x1, x2 = temp[0], temp[1]
    a_d, b_d = get_a_and_b(x1, x2)
    # g = a_d * x + b_d
    e_out += np.sum((a_d * pop + b_d - pop ** 2) ** 2) / num_pop
e_out /= itr
print('e_out', e_out)


# Calculate bias
bias = np.sum((pop * a + b - pop ** 2) ** 2) / num_pop
print('bias', bias)


# Calculate var
var = 0
for i in range(itr):
    #generate random D
    temp = random_pop(2)
    x1, x2 = temp[0], temp[1]
    a_d, b_d = get_a_and_b(x1, x2)
    var += np.sum((pop * a_d + b_d - pop * a - b) ** 2) / num_pop
var /= itr
print('var', var)


# plot
import matplotlib.pyplot as plt
x = np.linspace(-1, 1, 100)
plt.plot(x, x * a + b)
plt.plot(x, x ** 2)
plt.show()