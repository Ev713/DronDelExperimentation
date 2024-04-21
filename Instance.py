import itertools
import random
import copy
import sys
import warnings
from typing import List, Dict, Optional

import Agent
import State
import Vertex


def one_val_per_key_combinations(dict: Dict):
    """
    :param dict: {a1: [v1_1, v2_1, ... , v_n_1], a2: [v1_2, v2_2, ... , v_n_2], ... }
    :return: [{a1:v1_1, a2: v1_2, ...}, {a1:v2_1, a2: v1_2, ...},{,a1:v1_1, a2: v2_2, ...} {a1:v2_1, a2: v2_2, ...}...]
    """
    mini_dicts = {key: [] for key in dict}
    for key in dict:
        for l in dict[key]:
            mini_dicts[key].append({key: l})
    big_dict = [combo for combo in itertools.product(*mini_dicts.values())]
    combinations = []
    for a in big_dict:
        a_dict = {}
        for mini_dict in a:
            a_dict.update(mini_dict)
        combinations.append(a_dict)
    return combinations


class Instance:
    def __init__(self, name: str, map: List[Vertex.Vertex], agents: List[Agent.Agent], horizon: int, source: str = '-'):
        self.name = name
        self.type = self.name.split('_')[-2]
        self.map:List[Vertex.Vertex] = map  # list of Vertices
        if not self.sum_of_probs_is_1():
            for v in self.map:
                sum_of_probs = sum(v.distribution.values())
                if sum_of_probs == 0:
                    v.distribution = {0: 1.0}
                else:
                    v.distribution = {r: round(v.distribution[r] / sum_of_probs, 5) for r in v.distribution}
                    v.distribution[0] = 1 - (sum(list(v.distribution.values())) - v.distribution[0])
        self.map_map: Dict[int, Vertex.Vertex] = {v.hash(): v for v in map}
        self.agents: List[Agent] = agents  # list of agents
        self.agents_map: Dict[int, Agent.Agent] = {a.hash(): a for a in agents}
        self.horizon: int = horizon
        self.initial_state = (agents.copy(), map.copy())
        self.dropoffs: bool = True
        # self.check_sums_of_probs_is_0()
        self.distance: Dict[int, int] = {}
        self.source = source

    def get_time(self, state: State):
        return self.horizon - state.time_left

    def sum_of_probs_is_1(self):
        for v in self.map:
            if 0 not in v.distribution:
                v.distribution[0] = 0
            if round(sum(v.distribution.values()), 7) != 1:
                warnings.warn("Sum of probabilities in instance " + self.name + " vertex " + str(v) +
                              " is not 1!\nThe Distribution will be fixed.",
                              UserWarning)

                return False
        return True

    def actions(self, state: State, path: Dict[int, List[State.Action]]):
        # action: {a1: v_k, a2: v_m, ...  }
        agent_actions = {}
        # time = len(state.path[list(state.path.keys())[0]]) - state.time_left - 1
        time = self.horizon - state.time_left
        path_hash = {a: tuple((act.hash() for act in path[a])) for a in path}
        for a_hash in self.agents_map:
            a_loc = state.get_loc(a_hash)
            if self.agents_map[a_hash].movement_budget <= time:
                agent_actions[a_hash] = [State.Action(a_loc, True)]
            else:
                dests = [a_loc] + [n for n in self.map_map[a_loc].neighbours]
                agent_actions[a_hash] = [State.Action(n, True) for n in dests if
                                         State.Action(n, True).hash() not in path_hash[a_hash]] + \
                                        [State.Action(n, False) for n in dests if self.dropoffs]
        actions = [a for a in one_val_per_key_combinations(agent_actions)]
        return actions

    def map_is_connected(self):
        connected = [self.map[0]]
        while True:
            no_new_vertices = True
            for v in connected:
                for n in v.neighbours:
                    if n not in connected:
                        connected.append(n)
                        no_new_vertices = False
            if no_new_vertices:
                break
        return len(self.map) == len(connected)

    def make_det_map_and_det_map_map(self):
        det_map = []
        det_map_map = {}
        for v in self.map:
            det_v = Vertex.EmpVertex(v.hash())
            det_map.append(det_v)
            det_map_map[v.hash()] = det_v
        for v in self.map:
            det_v = det_map_map[v.hash()]
            det_v.neighbours = v.neighbours.copy()
        return det_map, det_map_map

    def make_special_map_and_map_map(self, ver_builder):
        map = []
        map_map = {}
        for v in self.map:
            new_v = ver_builder(v.hash())
            map.append(new_v)
            map_map[v.hash()] = new_v
            new_v.distribution = copy.deepcopy(v.distribution)
            new_v.neighbours = v.neighbours
        return map, map_map

    def make_agents_and_agents_map(self, agent_builder):
        new_agents = []
        new_agents_map = {}
        for a in self.agents:
            new_a = agent_builder(a.id, a.loc, a.movement_budget, a.utility_budget)
            new_agents.append(new_a)
            new_agents_map[a.hash()] = new_a
        return new_agents, new_agents_map

    def check_agents_integrity(self):
        for a in self.agents:
            if a not in self.agents_map.values():
                return False
        for a_hash in self.agents_map:
            if self.agents_map[a_hash] not in self.agents:
                return False
        return True

    def make_action(self, action: List[State.Action], state: State):  # Abstract method
        pass
