
import math

import random


dimension = 3
size = 2

# direct : value
spin_dic = {0: -1, 1: 1}


class Spin(object):
    def __init__(self, direct):
        self.direct = direct
        self.value = spin_dic[direct]

    def flip(self):
        # should be modified later
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


def get_random_position():
    pos = []
    for _ in range(dimension):
        pos.append(random.randint(0, size - 1))
    return pos


def get_spin_by_lattice_position(sigma, pos):
    getter = sigma
    for i in range(dimension):
        getter = getter[pos[i]]
    return getter


def calc_lattice_energy(sigma, pos):
    nears = nearby(pos)
    energy = 0
    for i in range(len(nears)):
        energy = energy + get_spin_by_lattice_position(sigma, nears[i]).value
    energy *= -1 * get_spin_by_lattice_position(sigma, pos).value
    return energy


def metropolis(sigma, T):
    pos = get_random_position()
    de = -2 * calc_lattice_energy(sigma, pos)
    if (de < 0) or (random.random() < math.exp(-de/T)):
        get_spin_by_lattice_position(sigma, pos).flip()
        return de, pos, True
    return de, pos, False


def get_positions():
    res = []

    def traverse(pos):
        for x in range(size):
            pos.append(x)
            if len(pos) == dimension:
                res.append(pos.copy())
            else:
                traverse(pos)
            pos.pop()
    traverse([])
    return res


def calc_total_magnetization(sigma):
    m = 0
    position = get_positions()
    for pos in position:
        m += get_spin_by_lattice_position(sigma, pos).value
    return m


def calc_total_energy(sigma):
    e = 0
    position = get_positions()
    for pos in position:
        e += calc_lattice_energy(sigma, pos)
    return e


def main_loops():
    n = size ** dimension
    monte_carlo_steps = 10000;
    transient = 1000
    norm = (1.0 / float(monte_carlo_steps * n));
    E = 0; Esq = 0; Esq_avg = 0; E_avg = 0; etot = 0; etotsq = 0
    M = 0; Msq = 0; Msq_avg = 0; M_avg = 0; mtot = 0; mtotsq = 0
    Mabs = 0; Mabs_avg = 0; mabstot = 0; mqtot = 0
    T = 5.0; minT = 0.5; deltT = 0.1

    ft = open('temperature.txt', 'w')
    feavg = open('E_avg.txt', 'w'); fesqavg = open('Esq_avg.txt', 'w')
    fmavg = open('M_avg.txt', 'w'); fmsqavg = open('Msq_avg.txt', 'w')
    fmabsavg = open('Mabs_avg.txt', 'w')

    sigma = initialization(dimension)
    while T >= minT:
        # Transient Function
        for _ in range(transient):
            for _ in range(n):
                metropolis(sigma, T)
        M = calc_total_magnetization(sigma)
        Mabs = abs(M)
        E = calc_total_energy(sigma)

        # Initialize summation variables at each temperature step
        etot = 0; etotsq = 0; mtot = 0; mtotsq = 0; mabstot = 0; mqtot = 0

        # Monte Carlo loop
        for _ in range(monte_carlo_steps):
            for _ in range(n):
                de, pos, flip = metropolis(sigma, T)
                if flip:
                    E += 2 * de
                    v = get_spin_by_lattice_position(sigma, pos).value
                    M += 2 * v
                    Mabs += abs(2 * v)

            # Keep summation of observables
            etot += E / 2.0   # so as not to count the energy for each spin twice
            etotsq += E / 2.0 * E / 2.0
            mtot += M
            mtotsq += M * M
            mqtot += M * M * M * M
            mabstot += (math.sqrt(M * M))

        # Average observables
        E_avg = etot * norm; Esq_avg = etotsq * norm;
        M_avg = mtot * norm; Msq_avg = mtotsq * norm; Mabs_avg = mabstot * norm; Mq_avg = mqtot * norm;

        # output
        st = "{:.9f}".format(T) + '\n'; ft.write(st)
        seavg = "{:.9f}".format(E_avg) + '\n'; feavg.write(seavg);
        sesqavg = "{:.9f}".format(Esq_avg) + '\n'; fesqavg.write(sesqavg)
        smavg = "{:.9f}".format(M_avg) + '\n'; fmavg.write(smavg);
        smsqavg = "{:.9f}".format(Msq_avg) + '\n'; fmsqavg.write(smsqavg)
        smabsavg = "{:.9f}".format(Mabs_avg) + '\n'; fmabsavg.write(smabsavg)

        print("{:.3f}".format(T))
        T -= deltT

    print("Finished")
    ft.close()
    feavg.close(); fesqavg.close()
    fmavg.close(); fmsqavg.close()
    fmabsavg.close()


if __name__ == '__main__':
    main_loops()
