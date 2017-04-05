import random
import math

dimension = 3
size = 2

# direct : value
spin_dic = {0: -1, 1: 1}


class Spin(object):
    def __init__(self, direct):
        self.direct = direct
        self.value = spin_dic[direct]
    def flap(self):
        self.direct = 1 - self.direct
        self.value = spin_dic[self.direct]


def getRandomSpin():
    ran = random.randint(0, len(spin_dic) - 1)
    return Spin(ran)


def initialization(d):
    if d == 1:
        sigma_1 = []
        for x in range(size):
            sigma_1.append(getRandomSpin())
        return sigma_1
    else:
        sigma_d = []
        for x in range(size):
            sigma_d.append(initialization(d - 1))
    return sigma_d


def getOutput(sigma):
    res = sigma
    for x in range(dimension - 1):
        res = sum(res, [])
    return [spin.direct for spin in res]


def nearby(pos):
    nears = []
    for i in range(len(pos)):
        left = pos.copy()
        left[i] = left[i] - 1
        if left[i] < 0:
            left[i] = size - 1
        right = pos.copy()
        right[i] = right[i] + 1
        if right[i] == size:
            right[i] = 0
        nears.append(left)
        nears.append(right)
    return nears


def random_position():
    pos = []
    for i in range(dimension):
        pos.append(random(0, size - 1))
    return pos


def get_spin_by_position(pos):
    getter = sigma
    for i in range(dimension):
        getter = getter[pos[i]]
    return getter


def calc_energy(pos):
    nears = nearby(pos)
    energy = 0
    for i in range(len(nears)):
        energy = energy + get_spin_by_position(nears[i]).value
    energy = -1 * get_spin_by_position(pos).value * energy
    return energy


def metropolis_loop(T):
    pos = random_position()
    de = -2 * calc_energy(pos)
    if (de < 0) or (random.random() < math.exp(-de/T)):
        get_spin_by_position(pos).flap()


if __name__ == '__main__':
    for i in range(5):
        sigma = initialization(dimension)
        res = getOutput(sigma)

        # print(getOutput(sigma))
        # print(getOutput(sigma))
