import numpy as np
from numba import njit


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
