from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import art
import data
import tricks


def mulTable(lst: list[np.ndarray]) -> (np.ndarray, np.ndarray):
    A = np.asarray(lst, dtype=np.complex64)
    T = np.einsum('mij,njk->mnik', A, A).astype(np.complex64)
    arr, table = tricks._mul_table_aux(A, T)
    return arr, table


def makeFiniteGroup(generators: list[np.ndarray]):
    elements = list(map(np.asarray, generators))
    n = len(elements)
    while True:
        elements, table = mulTable(elements)
        m = len(elements)
        print("Finding multiplication table. Current number of elements:", m)
        if m == n:
            break
        else:
            n = m
    return FiniteGroup(elements, table)


class Element:
    def __init__(self, group, idx):
        self.group = group
        self.idx = idx[0] if isinstance(idx, Iterable) else idx

    def __mul__(self, o: 'Element') -> 'Element':
        return self.group[self.group.table[self.idx, o.idx]]

    def __repr__(self):
        return str(self.group.matrix_elements[self.idx])

    @property
    def inv(self):
        return self.group[np.where(self.group.rmul(self.idx) == 0)[0]]


class FiniteGroup:
    def __init__(self, matrix_elements: np.ndarray, table: np.ndarray = None):
        self.matrix_elements = matrix_elements
        if table is None:
            _, self.table = mulTable(matrix_elements.tolist())
        else:
            self.table = table
        self.permutation = list(range(len(self)))

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        return Element(self, idx)

    @property
    def Table(self):
        return self.subTable(self.permutation)

    def lmul(self, g) -> np.ndarray:
        """
        :param g: int (an element) or list[int] (elements)
        :return: G.g
        """
        return self.table[:, g]

    def rmul(self, g) -> np.ndarray:
        """
        :param g: int (an element) or list[int] (elements)
        :return: g.G
        """
        return self.table[g, :]

    def subgroup_lmul(self, indices, g) -> np.ndarray:
        return self.table[:, indices][g, :]

    def subgroup_rmul(self, indices, g) -> np.ndarray:
        return self.table[indices, :][:, g]

    def subTable(self, indices: list[int]) -> np.ndarray:
        return self.table[indices, :][:, indices]

    def subgroup(self, indices: list[int]):
        sub_table = self.subTable(indices)
        if set(sub_table.reshape(-1).tolist()) == set(indices):
            return FiniteGroup(self.matrix_elements[indices])
        else:
            return None

    def invariantSubgroup(self, indices: list[int]):
        for g in range(len(self)):
            if set(self.table[indices, g]) != set(self.table[g, indices]):
                return None
        return self.subgroup(indices)

    def cosets(self, indices: list[int]) -> np.ndarray:
        # return: a 2d matrix, each line is a coset (represented by indices)
        assert self.subgroup(indices) is not None
        res = []
        for g in range(len(self)):
            coset_g = set(self.subgroup_rmul(indices, g))
            if coset_g not in res:
                res.append(coset_g)
        return np.array(list(map(lambda x: np.sort(list(x)), res)))

    def sortByCoset(self, coset: np.ndarray):
        self.permutation = coset.reshape(-1)
        return self

    def groupByCoset(self, indices: list[int]):
        return self.sortByCoset(self.cosets(indices)).rename()

    def quotient(self, indices: list[int]) -> 'FiniteGroup':
        assert self.invariantSubgroup(indices) is not None
        n = len(indices)
        m = len(self) // n
        coset = self.cosets(indices)
        self.sortByCoset(coset)
        table = self.Table[::n, ::n]
        res = np.zeros_like(table)
        for i in range(m):
            for j in range(n):
                res[table == coset[i, j]] = i
        return FiniteGroup(self.matrix_elements[coset[:, 0], :, :], res)

    def centralizer(self, g=None):
        """
        :param g: int (an element) or list[int] (elements)
        :return: C_G(g), the centralizer of g in group G,
        defined as {a in G | ag = ga}
        """
        if g is None:
            g = list(range(len(self)))
        gG = self.rmul(g)
        Gg = self.lmul(g).T
        if isinstance(g, Iterable):
            return [np.where(gG[i] == Gg[i])[0] for i in range(len(g))]
        else:
            return np.where(gG == Gg)[0]

    def conjugateClassOf(self, g: int) -> list[int]:
        cent = self.centralizer(g)
        H = self.cosets(cent)
        return [(self[H[i, 0]] * self[g] * self[H[i, 0]].inv).idx for i in range(H.shape[0])]

    def conjugateClasses(self):
        def flatten(llst):
            return [i for lst in llst for i in lst]

        lst = []
        for g in range(len(self)):
            if g in flatten(lst): continue
            lst.append(self.conjugateClassOf(g))
        lst.sort(key=len)
        return lst

    def rename(self):
        res = np.zeros_like(self.table)
        for i in range(len(self)):
            res[self.Table == self.permutation[i]] = i
        self.matrix_elements = self.matrix_elements[self.permutation]
        self.table = res
        self.permutation = list(range(len(self)))
        return self

    def show(self):
        n = np.max(self.table)
        colors = art.getColors(n)
        cmap = ListedColormap(colors)
        plt.imshow(self.Table, cmap=cmap)
        plt.show()


if __name__ == '__main__':
    G = makeFiniteGroup(data.ex3)
    print(G.conjugateClasses())
