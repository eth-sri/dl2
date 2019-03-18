import numpy as np

class Graph:

    def __init__(self, n, m, adj):
        self.n = n
        self.m = m
        self.adj = adj
        self.inf = 10000000000

    def add_edge(self, a, b):
        self.adj[(a,b)] = 1
        self.adj[(b,a)] = 1

    def construction_finished(self):
        self.x, self.y = [], []
        for (i, j) in self.adj:
            self.x.append(i)
            self.y.append(j)
        self.get_paths()

    def get_paths(self):
        self.p = -np.ones((self.n, self.n))

        for i in range(self.n):
            self.p[i,i] = 0.0
        for (a, b) in self.adj:
            self.p[a,b] = 1.0

        for z in range(self.n):
            for x in range(self.n):
                for y in range(self.n):
                    if self.p[x,z] < 0 or self.p[z,y] < 0:
                        continue
                    if self.p[x,y] < 0:
                        self.p[x,y] = self.p[x,z] + self.p[z,y]
                    else:
                        self.p[x,y] = np.minimum(self.p[x,y], self.p[x,z] + self.p[z,y])

    @staticmethod
    def gen_random_graph(n, m):
        retg = Graph(n, m, {})

        assert m >= n - 1
        for j in range(1,n):
            i = np.random.randint(j)
            retg.add_edge(i, j)

        for i in range(m - (n-1)):
            i, j = 0, 0
            while i == j or (i, j) in retg.adj:
                i = np.random.randint(n)
                j = np.random.randint(n)
            retg.add_edge(i, j)

        retg.construction_finished()
        return retg
