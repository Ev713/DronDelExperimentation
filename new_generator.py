import warnings

import Agent
import Instance
import Vertex
import numpy as np


class Generator:
    def __init__(self, rows, cols, hor, unpassable_locs=None, move_budgets=None, util_budgets=None, init_locs=None,
                 distribs=None):
        self.rows = rows
        self.hor = hor
        self.cols = cols
        if unpassable_locs is None:
            self.unpassable_locs = []
        else:
            self.unpassable_locs = unpassable_locs
        if util_budgets is None:
            self.util_budgets = []
        else:
            self.util_budgets = util_budgets
        if move_budgets is None:
            self.move_budgets = []
        else:
            self.move_budgets = move_budgets

        self.locs = [(x, y) for x in range(self.cols) for y in range(self.rows) if (x, y) not in self.unpassable_locs]
        if init_locs is None:
            self.init_locs = [self.locs[0] for _ in self.util_budgets]
        else:
            self.init_locs = init_locs

        if distribs is None:
            self.distribs = {loc: {0: 1} for loc in self.locs}
        else:
            self.distribs = distribs
            if not all([loc in distribs for loc in self.locs]):
                warnings.warn("Insufficient distributions, ones that are not indicated will be replaced with {0:1}",
                              UserWarning)
                for loc in self.locs:
                    if loc not in distribs:
                        distribs[loc] = {0: 1}

        if len(move_budgets) != len(util_budgets) or len(util_budgets) != len(init_locs):
            raise Exception('Due to incoherent lengths of inputs, number of agents is unclear')

    def to_picture(self):
        colors = [[(0, 0, 0) for _ in self.cols ] for _ in self.rows]
        for loc in self.locs:
            p = 1-self.distribs[loc][0]
            r = sum()

    def generate_inst(self):
        locs = self.locs
        vs = []
        for loc in locs:
            v = Vertex.Vertex(locs.index(loc))
            v.distribution = self.distribs[loc]
            v.neighbours = [locs.index((x, y)) for (x, y) in
                            [(loc[0] + 1, loc[1]), (loc[0] - 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)]
                            if (x, y) in locs]
            vs.append(v)
        agents = []
        for a in range(len(self.util_budgets)):
            agents.append(Agent.Agent(a, self.locs.index(self.init_locs[a]),
                                      self.move_budgets[a], self.util_budgets[a]))
        return Instance.Instance(f'i_{len(vs)}_{len(agents)}_{self.hor}', vs, agents, self.hor)


class AntiGreedyGenerator(Generator):
    def __init__(self, size, p):
        super().__init__(size, size, size, move_budgets=[size], util_budgets=[size], init_locs=[(0, 0)],
                         unpassable_locs=[(x, y) for x in range(1, size) for y in range(1, size)])
        self.p = p
        self.distribs = {}
        for loc in self.locs:
            x = loc[0]
            y = loc[1]
            if y == 0:
                self. distribs[loc] = {1: 1}
            if x == self.cols - 1:
                size = self.cols
                R = np.ceil((size + 1) * size / self.p)
                d = {r: 0 for r in range(R)}
                d[R] = self.p
                self. distribs[loc] = d


class MountainTopGenerator(Generator):

    def __init__(self, rows, cols, hor, unpassable_locs=None, move_budgets=None, util_budgets=None, init_locs=None,
                 center_locs=None, decrease=0.25):
        super().__init__(rows, cols, hor, unpassable_locs, move_budgets, util_budgets, init_locs)
        if center_locs is None:
            self.center_locs = []
        else:
            self.center_locs = center_locs
        self.decrease = decrease
        for loc in self.locs:
            x = loc[0]
            y = loc[1]
            dist = min([abs(x - cx) + abs(y - cy) for cx, cy in self.center_locs])
            self.distribs[loc] = {0: 1 - max(0, 1 - self.decrease * dist), 1: max(0, 1 - self.decrease * dist)}


class FullRandomGenerator(Generator):
    def __init__(self, rows, cols, hor, unpassable_locs=None, move_budgets=None, util_budgets=None, init_locs=None,
                 max_r=7):
        super().__init__(rows, cols, hor, unpassable_locs, move_budgets, util_budgets, init_locs)
        for loc in self.locs:
            ps = np.array([np.random.random() for _ in range(max_r)])
            ps /= sum(ps)
            self.distribs[loc] = {i: ps[i] for i in range(max_r)}


class SanityCheckGenerator(Generator):
    def __init__(self, size):
        super().__init__(size, size, size * 2 - 1, move_budgets=[size * 2 - 1, size * 2 - 1],
                         util_budgets=[size * 2 - 1, size * 2 - 1], init_locs=[(0, 0), (0, 0)])
        self.distribs = {loc: {0: 0, 1: 1} if
        ((loc[0] == 0) != (loc[1] == 0)) or ((loc[0] == size - 1) != (loc[1] == size - 1)) else {0: 1}
                         for loc in self.locs}


mt = MountainTopGenerator(5, 5, 15, [(1, 2)], [14, 13], [6, 8], [(0, 0), (0, 0)],
                          [(2, 2), (3, 1), (4, 4)]).generate_inst()
ag = AntiGreedyGenerator(20, 0.0001)
sc = SanityCheckGenerator(6)
fr = FullRandomGenerator(5, 5, 15, [(1, 2)], [14, 13], [6, 8], [(0, 0), (0, 0)], 7)
breakpoint()
