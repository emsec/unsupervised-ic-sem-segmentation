class UnionFind:
    def __init__(self, num):
        self.parents = [i for i in range(num)]
        self.rank = [0] * num

    def find(self, x):
        root = x
        parent = self.parents[root]
        while parent != root:
            root = parent
            parent = self.parents[root]

        parent = self.parents[x]
        while parent != root:
            self.parents[x] = root
            x = parent
            parent = self.parents[x]

        return root

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x != y:
            if self.rank[x] < self.rank[y]:
                x, y = y, x
            # x has larger rank and becomes shared root
            self.parents[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1


if __name__ == '__main__':
    dut = UnionFind(5)
    dut.union(0, 1)
    dut.union(1, 2)
    dut.union(3, 4)
    assert dut.find(0) == dut.find(2)
    assert dut.find(0) == dut.find(1)
    assert dut.find(1) == dut.find(2)

    assert dut.find(4) == dut.find(3)
    assert dut.find(1) != dut.find(3)
