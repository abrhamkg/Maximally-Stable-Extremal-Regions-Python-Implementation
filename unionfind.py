import numpy as np


class CompHistory(object):
    instances = []
    count = 0

    def __init__(self, size=1, members=None, delta=2, children_sizes=None, parent_sizes=None, neighbors=None):
        if members is None:
            members = set()

        if children_sizes is None:
            self.children_sizes = [0] * delta
        else:
            assert len(children_sizes) == delta
            self.children_sizes = children_sizes

        if parent_sizes is None:
            self.parent_sizes = [0] * delta
        else:
            assert len(parent_sizes) == delta
            self.parent_sizes = parent_sizes
        if neighbors is None:
            neighbors = set()

        self.delta = delta
        self.size = size
        self.name = CompHistory.count
        CompHistory.instances.append(self)
        self.toplevel = True
        self.parent = None
        self.right = None
        self.left = None
        self.members = members
        self.neighbors = neighbors

    def add_neighbor(self, entry):
        self.neighbors.add(entry)

    def set_left(self, other):
        self.left = other

    def set_right(self, other):
        self.right = other

    def __add__(self, other):
        # Determine the size of the new(parent) connected componenet
        size = self.size + other.size
        # Merged components are no longer toplevel
        self.toplevel = False
        other.toplevel = False
        # Determine children components history for the new(parent) component
        operands = [self, other]
        max_size, idx = max((self.size, 0), (other.size, 1))
        larger = operands[idx]
        children_sizes = [larger.children_sizes[-1], larger.size]

        # Join the components
        members = self.members.union(other.members)
        neighbors = self.neighbors.union(other.neighbors)
        neighbors = neighbors.difference(members)
        history = CompHistory(size, members=members, children_sizes=children_sizes, neighbors=neighbors)
        history.left = self
        history.right = other
        return history

    def __str__(self):
        return "CompHistory(size={}, toplevel={})".format(self.size, self.toplevel)

    def __repr__(self):
        return "CompHistory(size={}, toplevel={})".format(self.size, self.toplevel)


class UnionFind(object):
    def __init__(self, num_sites, delta=2):
        self.__delta = delta
        self.__N = num_sites
        self.__componenet = np.arange(num_sites) # [-1] * self.__N #
        self.__count = num_sites
        self.__weight = np.ones_like(self.__componenet)
        self.__history = {i:CompHistory(size=1, members={i,}) for i in range(num_sites)}

    def add_history(self, id_list):
        for site_id in id_list:
            self.__history[site_id] = CompHistory()

    def find(self, p):
        parent = self.__componenet[p]
        while parent != p:
            # Path compression
            self.__componenet[p] = parent
            p = parent
            parent = self.__componenet[p]
        return parent

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root == q_root:
            return
        if self.__weight[p_root] > self.__weight[q_root]:
            p_root, q_root = q_root, p_root

        self.__componenet[p_root] = q_root
        self.__weight[q_root] += self.__weight[p_root]
        self.__history[q_root] = self.__history[q_root] + self.__history[p_root]
        self.__weight[p_root] = 0
        self.__count -= 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def count(self):
        return self.__count

    def componenets(self):
        pass#all_components

    def get_top_level_history(self):
        return [history for history in self.__history.values() if history.toplevel]

if __name__ == '__main__':
    with open('data/uf/input.txt') as f:
        lines = f.readlines()
        num_sites, lines = int(lines[0].strip("\n")), lines[1:]
        a = UnionFind(num_sites)
        for line in lines:
            p, q = line.strip("\n").split()
            p, q = int(p), int(q)
            a.union(p, q)
    print(a.count())
    history = a.get_top_level_history()
    print(history)
    b = history[0]

    # print(a.count())
    # a.union(0, 1)
    # print(a.count(), a.find(0), a.find(1))
    # a.union(1, 0)
    # print(a.count(), a.find(0), a.find(1))
    # a.union(2, 0)
    # print(a.count(), a.find(0), a.find(1))


