import math
import random
import warnings
import numpy as np

import Agent
import Instance
import Vertex


class ConnectedComponentsException(Exception):
    pass


class Generator:
    def __init__(self, gen_type, cols, rows, num_of_agents, horizon, ag_p=None, source='X', name=None, unpassable=None):
        self.file_name = None
        self.name = name
        self.max_reward = 7
        self.source = source
        self.cols = cols
        self.rows = rows
        self.num_of_agents = num_of_agents
        self.horizon = horizon
        self.prec = 5  # precision
        self.type = gen_type
        self.unpassable = unpassable if unpassable else []

        self.initialize_generator(ag_p)
        self.gen_name(name)

    def initialize_generator(self, ag_p):
        if self.type == 'MT':
            self.num_of_centers = self.generate_num_of_centers()
            self.decrease = 0.25
            self.generate_centers_and_distances()
        elif self.type == 'AG':
            self.validate_ag_params(ag_p)
            self.set_anti_greedy_defaults()
        elif self.type == 'SC':
            self.validate_sc_params()
        # Other types don't need initialization

    def validate_ag_params(self, ag_p):
        if ag_p is None:
            raise ValueError("Type is anti-greedy but no probability parameter is given.")
        self.ag_p = ag_p
        if self.source != 'X':
            raise ValueError("Parsing into AG is not supported.")
        if self.horizon != self.rows:
            self.horizon = self.rows
            warnings.warn("Horizon was adjusted to be equal to number of rows.", UserWarning)
        if self.max_reward != math.ceil(self.cols / self.ag_p):
            self.max_reward = math.ceil(self.cols / self.ag_p)
            warnings.warn("Maximum reward was adjusted.", UserWarning)
        self.unpassable = [self.enhash(x, y) for x in range(self.rows) for y in range(self.cols) if x != 0 and y != 0]

    def validate_sc_params(self):
        if self.horizon != self.rows + self.cols - 3:
            self.horizon = self.rows + self.cols - 3
            warnings.warn("Horizon was adjusted for sanity-check.", UserWarning)
        if self.num_of_agents != 2:
            self.num_of_agents = 2
            warnings.warn("Number of agents was adjusted for sanity-check.", UserWarning)

    def set_anti_greedy_defaults(self):
        if self.source != 'X':
            raise ValueError("Parsing into AG is not supported.")
        if self.horizon != self.rows:
            self.horizon = self.rows
            warnings.warn("Horizon was adjusted to be equal to number of rows.", UserWarning)
        if self.max_reward != math.ceil(self.cols / self.ag_p):
            self.max_reward = math.ceil(self.cols / self.ag_p)
            warnings.warn("Maximum reward was adjusted for anti-greedy.", UserWarning)
        self.unpassable = [self.enhash(x, y) for x in range(self.rows) for y in range(self.cols) if x != 0 and y != 0]

    def generate_centers_and_distances(self):
        self.centers = self.generate_centers()
        self.dist_to_center = self.generate_distances(self.centers)

    def generate_num_of_centers(self):
        return math.ceil(self.get_map_size() / 20)

    def get_map_size(self):
        return self.rows * self.cols - len(self.unpassable)

    def gen_name(self, name=None):
        if name is None:
            type_suffix = self.type if self.type != 'AG' else 'AG' + "".join(str(self.ag_p).split('.'))
            self.name = f'i_{self.get_map_size()}_{self.num_of_agents}_{self.horizon}_{type_suffix}_{self.source}'
        else:
            self.name = name
        self.file_name = self.name

    def get_vertices(self):
        return [(x, y) for x in range(self.cols) for y in range(self.rows) if self.xy_is_legal(x, y)]

    def generate_init_loc_hash(self, agent_hash):
        if self.type == 'FL':
            return self.enhash(self.rows // 2, self.cols // 2)
        vertices = self.get_vertices()
        if not vertices:
            raise ValueError("No legal squares")
        return vertices[0]

    def generate_full_random_distr(self, vertex_hash):
        distr_size = random.randint(2, self.max_reward + 1) if self.max_reward > 1 else random.randint(1, 2)
        distr = {0: 1}
        for _ in range(distr_size - 1):
            r = np.random.randint(1, self.max_reward)
            distr[r] = round(random.uniform(pow(1 / 10, self.prec), 1 - sum(distr.values())), self.prec)
        distr[0] = round(1 - sum(distr.values()), self.prec)
        return distr

    def generate_empty_distr(self):
        return {0: 1}

    def generate_centers(self):
        centers = set()
        vertices = self.get_vertices()
        while len(centers) < self.num_of_centers:
            center = random.choice(vertices)
            centers.add(center)
        return centers

    def generate_distances(self, centers):
        distances = {}
        level = 0
        next_level = centers
        while len(distances) < self.get_map_size():
            for c in next_level:
                distances[c] = level
            new_level = set()
            for c in next_level:
                new_level.update(self.get_neighbours(c))
            new_level.difference_update(distances.keys())
            if not new_level:
                break
            next_level = new_level
            level += 1
        for v in self.get_vertices():
            if v not in distances:
                distances[v] = -1
        return distances

    def distance_to_center_to_distr(self, x):
        prob_of_zero = min(x * self.decrease, 1) if x != -1 else 1
        return {1: round(1 - prob_of_zero, self.prec), 0: round(prob_of_zero, self.prec)}

    def generate_mountain_top_distr(self, vertex_hash):
        if not hasattr(self, 'centers'):
            self.generate_centers_and_distances()
        return self.distance_to_center_to_distr(self.dist_to_center[vertex_hash])

    def generate_distr(self, v_hash):
        distr_generators = {
            'FR': self.generate_full_random_distr,
            'EMPTY': self.generate_empty_distr,
            'MT': self.generate_mountain_top_distr,
            'AG': self.generate_anti_greed_distr,
            'SC': self.generate_sanity_check_distr,
            'FL': self.generate_sanity_check_distr
        }
        distr = distr_generators.get(self.type, lambda x: {})(v_hash)
        if self.distr_is_legal(distr):
            return distr
        fixed_distr = self.fixed_distr(distr)
        warnings.warn(f"Illegal distribution in: {self.name}\nOriginal: {distr}\nFixed: {fixed_distr}", UserWarning)
        return fixed_distr

    def fixed_distr(self, distr):
        distr_sum = sum(distr.values())
        return {r: value / distr_sum for r, value in distr.items()} if distr_sum else {0: 1}

    def distr_is_legal(self, distr):
        return round(sum(distr.values()), self.prec) == 1

    def generate_sanity_check_distr(self, v):
        x, y = self.dehash(*v)
        return {1: 1, 0: 0} if (x == 0 or x == self.cols - 1 or y == 0 or y == self.rows - 1) else {0: 1}

    def generate_anti_greed_distr(self, v):
        x, y = self.dehash(*v)
        if x == 0:
            return {math.ceil(self.cols / self.ag_p): self.ag_p, 0: 1 - self.ag_p} if y == self.rows - 1 else {0: 1}
        return {1: 1, 0: 0} if y == 0 else {0: 1}

    def generate_utility_budget(self, agent_hash):
        return max(random.randint(1, self.horizon), 3)

    def generate_movement_budget(self, agent_hash):
        if self.type in ['AG', 'SC']:
            return self.horizon
        return random.randint(2, self.horizon) if self.horizon > 2 else 2

    def enhash(self, x, y):
        return self.get_vertices().index((x, y))

    def get_neighbours(self, x, y):
        potential_neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [self.enhash(m, n) for (m, n) in potential_neighbours if self.xy_is_legal(m, n)]

    def num_is_legal(self, num):
        return 0 < num <= self.cols * self.rows and num not in self.unpassable

    def xy_is_legal(self, x, y):
        if self.unpassable is None:
            raise ValueError("Unpassable list is not defined")
        return 0 <= x < self.cols and 0 <= y < self.rows and self.enhash(x, y) not in self.unpassable

    def dehash(self, id):
        return self.get_vertices()[id]

    def gen_instance(self):
        map_list = []
        map_map = {}
        neighbours_hashes = {}
        for y in range(self.rows):
            for x in range(self.cols):
                vertex_hash = self.enhash(x, y)
                if vertex_hash in self.unpassable:
                    continue
                v = Vertex.Vertex(*vertex_hash)
                v.distribution = self.generate_distr(vertex_hash)
                neighbours_hashes[v] = self.get_neighbours(x, y)
                map_map[vertex_hash] = v
                map_list.append(v)
        for v in map_list:
            v.neighbours = [map_map[n_hash] for n_hash in neighbours_hashes[v]]
        agents = [
            Agent.Agent(a, map_map[self.generate_init_loc_hash(a)],
                        self.generate_movement_budget(a),
                        self.generate_utility_budget(a))
            for a in range(self.num_of_agents)
        ]
        return Instance.Instance(self.name, map_list, agents, self.horizon, self.source)

g = Generator('MT', 10, 10, 3, 15)
inst = g.gen_
