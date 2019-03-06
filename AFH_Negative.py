# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import linalg
from numpy import linalg as la
import itertools

class AFH_Negative(object):
    def __init__(self,N = 8, J = 1., g = 0.):
        self.N = N
        self.J = J
        self.g = g

    def state2int(self, state):
        out = 0
        for bit in state:
            out = (out << 1) | int(0.5 * (1 + bit))
        return out

    def Sp(self, i, stateA):
        if stateA[i] == -1:
            stateB = np.copy(stateA)
            stateB[i] = 1
            return 1, stateB
        else:
            return 0, stateA

    def Sm(self, i, stateA):
        if stateA[i] == 1:
            stateB = np.copy(stateA)
            stateB[i] = -1
            return 1, stateB
        else:
            return 0, stateA

    def Sz(self, i, stateA):
        return stateA[i] * 0.5

    def SpSm(self, i, j, stateA):
        found, stateB = self.Sm(j, stateA)
        if found == 1: found, stateB = self.Sp(i, stateB)

        if found == 1:
            return 1, stateB
        else:
            return 0, stateA

    def SmSp(self, i, j, stateA):
        found, stateB = self.Sp(j, stateA)
        if found == 1: found, stateB = self.Sm(i, stateB)

        if found == 1:
            return 1, stateB
        else:
            return 0, stateA

    def getH(self):
        states = np.array(list(itertools.product([-1, 1], repeat=self.N)))
        nstates = states.shape[0]
        basisdict = dict([(self.state2int(states[i]), i) for i in range(nstates)])
        #print('The reduced Fock space is of size: ' + str(nstates))

        L1, L2 = self.N, 1  # L1 is the number of bonds, and L2 sets the interaction range
        bnd = np.empty((0, 2), dtype=np.int)  # contains the bonds
        for si in range(self.N):
            sj = (si + 1) % L1 + L1 * int(si / L1)
            bnd = np.vstack((bnd, np.array([si, sj])))
            if L2 > 1:
                sj = (si + L1) % (L1 * L2)
                bnd = np.vstack((bnd, np.array([si, sj])))
        nb = bnd.shape[0]

        spidx = np.empty((0, 2), dtype=np.uint16)  # sparse index
        spval = np.array([], dtype=np.float)  # sparse value
        for i in range(nstates):
            stateA = states[i]
            for b in range(nb):
                si, sj = bnd[b]
                found, stateB = self.SpSm(si, sj, stateA)
                if found == 1:
                    j = basisdict[self.state2int(stateB)]
                    spidx = np.vstack((spidx, np.array([i, j])))
                    spval = np.append(spval, self.J * 0.5)

                found, stateB = self.SmSp(si, sj, stateA)
                if found == 1:
                    j = basisdict[self.state2int(stateB)]
                    spidx = np.vstack((spidx, np.array([i, j])))
                    spval = np.append(spval, self.J * 0.5)

                spidx = np.vstack((spidx, np.array([i, i])))
                spval = np.append(spval, self.J * self.Sz(si, stateA) * self.Sz(sj, stateA))

        H = sp.sparse.coo_matrix((spval, (spidx[:, 0], spidx[:, 1])), shape=(nstates, nstates),
                                 dtype=np.float).tocsr()  # self.Note that here we have to convert the matrix csr form, in order to use Theano
        E_exact, Psi_exact = sp.sparse.linalg.eigsh(H, which='SA', k=1)
        return nstates, states, H, E_exact[0], Psi_exact[:, 0]

