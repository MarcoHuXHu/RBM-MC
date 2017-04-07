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

    def flip(self):
        # should be modify in further
        self.direct = 1 - self.direct
        self.value = spin_dic[self.direct]


def get_random_spin():
    ran = random.randint(0, len(spin_dic) - 1)
    return Spin(ran)


def initialization(d):
    if d == 1:
        sigma_1 = []
        for _ in range(size):
            sigma_1.append(get_random_spin())
        return sigma_1
    else:
        sigma_d = []
        for _ in range(size):
            sigma_d.append(initialization(d - 1))
    return sigma_d


def get_output(sigma):
    result = sigma
    for _ in range(dimension - 1):
        result = sum(result, [])
    return [spin.direct for spin in result]


def nearby(pos):
    nears = []
    for i in range(len(pos)):
        left = pos.copy()
        left[i] -= 1
        if left[i] < 0:
            left[i] = size - 1
        right = pos.copy()
        right[i] += 1
        if right[i] == size:
            right[i] = 0
        nears.append(left)
        nears.append(right)
    return nears


def random_position():
    pos = []
    for _ in range(dimension):
        pos.append(random(0, size - 1))
    return pos


def get_spin_by_latticeposition(sigma, pos):
    getter = sigma
    for i in range(dimension):
        getter = getter[pos[i]]
    return getter


def calc_lattice_energy(sigma, pos):
    nears = nearby(pos)
    energy = 0
    for i in range(len(nears)):
        energy = energy + get_spin_by_latticeposition(sigma, nears[i]).value
    energy *= -1 * get_spin_by_latticeposition(sigma, pos).value
    return energy


def metropolis(sigma, T):
    pos = random_position()
    de = -2 * calc_lattice_energy(pos)
    if (de < 0) or (random.random() < math.exp(-de/T)):
        get_spin_by_latticeposition(sigma, pos).flap()
        return de
    return None


def main_loops():
    n = size ** dimension
    E = 0; Esq = 0; Esq_avg = 0; E_avg = 0; etot = 0; etotsq = 0
    M = 0; Msq = 0; Msq_avg = 0; M_avg = 0; mtot = 0; mtotsq = 0
    Mabs = 0; Mabs_avg = 0; mabstot = 0; mqtot = 0
    T = 5.0; minT = 0.5; deltT = 0.1
    monte_carlo_steps = 10000

    sigma = initialization(dimension)
    while (T >= minT):

        for _ in range(monte_carlo_steps):
            for _ in range(n):
                metropolis()

        T -= deltT


if __name__ == '__main__':
    main_loops()