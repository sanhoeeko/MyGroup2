import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit

import art


class DictList:
    def __init__(self):
        self.dic = {}
        self.mapping_list = []

    def append(self, key, value):
        if key not in self.dic.keys():
            self.dic[key] = [len(self)]
            self.mapping_list.append(value)
        else:
            self.dic[key].append(len(self))
            self.mapping_list.append(value)

    def tolist(self):
        return self.mapping_list

    def __len__(self):
        return len(self.mapping_list)

    def index(self, query_key, query_value) -> int:
        if query_key not in self.dic.keys():
            return -1
        else:
            for idx in self.dic[query_key]:
                if np.all(self.mapping_list[idx] == query_value):
                    return idx
            return -1


def _mul_table_aux_py(lst: np.ndarray, T: np.ndarray):
    n = lst.shape[0]
    trace_map = np.einsum('mnii->mn', T).astype(np.int32)
    traces = trace_map[0]  # assume that the first element is the identity element
    dic = DictList()
    for i in range(n):
        dic.append(traces[i], lst[i])
    table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            Tij = T[i, j, :, :]
            trTij = trace_map[i, j]
            k = dic.index(trTij, Tij)
            if k == -1:
                table[i, j] = len(dic)
                dic.append(trTij, Tij)
            else:
                table[i, j] = k
    return np.asarray(dic.tolist()), table


@njit
def _mul_table_aux(lst: np.ndarray[np.complex64], T: np.ndarray[np.complex64]):
    n = lst.shape[0]
    arr = np.zeros((65536, T.shape[2], T.shape[3]), dtype=np.complex64)
    arr[:lst.shape[0], :, :] = lst
    table = np.zeros((n, n), dtype=np.int32)
    current_length = n
    for i in range(n):
        for j in range(n):
            Tij = T[i, j, :, :]
            flag = 0
            for k in range(current_length):
                if np.all(Tij == arr[k, :, :]):
                    table[i, j] = k
                    flag = 1
                    break
            if flag == 0:
                table[i, j] = current_length
                arr[current_length, :, :] = Tij
                current_length += 1
    return arr[:current_length, :, :], table


def mulTable(lst: list[np.ndarray]) -> (np.ndarray, np.ndarray):
    A = np.asarray(lst, dtype=np.complex64)
    T = np.einsum('mij,njk->mnik', A, A).astype(np.complex64)
    arr, table = _mul_table_aux(A, T)
    return arr, table


def groupMulTable(lst: list[np.ndarray]):
    elements = lst
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
        self.idx = idx

    def __mul__(self, o: 'Element') -> 'Element':
        return self.group[self.group.table[self.idx, o.idx]]


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

    def coset(self, indices: list[int]) -> np.ndarray:
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
        return self.sortByCoset(self.coset(indices)).rename()

    def quotient(self, indices: list[int]) -> 'FiniteGroup':
        assert self.invariantSubgroup(indices) is not None
        n = len(indices)
        m = len(self) // n
        coset = self.coset(indices)
        self.sortByCoset(coset)
        table = self.Table[::n, ::n]
        res = np.zeros_like(table)
        for i in range(m):
            for j in range(n):
                res[table == coset[i, j]] = i
        return FiniteGroup(self.matrix_elements[coset[:, 0], :, :], res)

    def rename(self):  # bug
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


# G = groupMulTable(data.ex3)
# G.sortByCoset(G.coset([0, 1, 2, 3])).rename()
# G.show()
# Q = G.quotient([0, 1, 2, 3])
# Q.sortByCoset(Q.coset([0, 10, 12, 16, 20, 23])).rename()
# Q.show()

with open('ex4.pkl', 'rb') as f:
    G: FiniteGroup = pkl.load(f)
# H1 = G.invariantSubgroup([0, 1, 2, 3, 4, 5, 6, 7])
Q1 = G.quotient([0, 1, 2, 3, 4, 5, 6, 7])
# H2 = Q1.invariantSubgroup([0, 119, 140, 144, 154, 165, 177, 188])
Q2 = Q1.quotient([0, 119, 140, 144, 154, 165, 177, 188])
# H3 = Q2.subgroup([0, 1, 2, 3, 10, 11])
Q2.groupByCoset([0, 1, 2, 3, 10, 11]).show()
Q2.groupByCoset([0, 10, 11]).show()
