import copy
import typing
from typing import List, Dict, Optional
import numpy as np

from Vertex import Vertex


class Ber:
    def __init__(self, r: float, p:float):
        self.p = p
        self.r = r

    def q(self):
        return 1 - self.p

    def e(self):
        return self.p * self.r

    def hash(self):
        return self.p, self.r

    def __str__(self):
        return str(self.hash())


class Action:
    def __init__(self, loc: int, dropoff: bool):
        self.loc: int = loc
        self.dropoff: bool = dropoff

    def __str__(self):
        return "( " + str(self.loc) + ", " + str(self.dropoff) + " )"

    def __repr__(self):
        return str(self)

    def hash(self):
        return self.loc, self.dropoff


class State:
    # Info held by every node. The info is gathered using agents actions only therefore is deterministic.
    def __init__(self):
        self.time_left = 0  # int

    def is_terminal(self):
        return self.time_left == 0

    def get_loc(self, a_hash: int):
        pass

    def copy(self):
        pass


class EmpState(State):
    def __init__(self, instance=None):
        super().__init__()
        self.path: Dict[int, List[Optional[Action]]] = {}  # a.hash(): [act, act, -1, ... , -1] -
        # Required to count the reward of the state. Not needed in stochastic state.
        if instance is not None:
            self.time_left = instance.horizon
            for a in instance.agents:
                self.path[a.hash()] = [None for _ in range(instance.horizon + 1)]
                self.path[a.hash()][0] = Action(a.loc, False)

    def get_loc(self, a_hash: int):
        if self.path[a_hash][0] is None:
            return None
        if self.path[a_hash][-1] is not None:
            return self.path[a_hash][-1].loc
        first_none = 0
        while self.path[a_hash][first_none] is not None:
            first_none += 1
        return self.path[a_hash][first_none - 1].loc

    def copy(self):
        copy_state = EmpState()
        copy_state.path = copy.deepcopy(self.path)
        copy_state.time_left = self.time_left
        return copy_state

    def __str__(self):
        return str((self.path, self.time_left))


class VectorState(State):
    def __init__(self, instance=None):
        super().__init__()
        self.reward: Optional[float] = None
        if instance is not None:
            self.loc: Dict[int:int] = {a.hash(): a.loc for a in instance.agents}
            self.bers: Dict[int, Ber] = {v.hash(): Ber(v.bernoulli(), v.p()) for v in instance.map}
            self.utl: Dict[int, List[float]] = {a.hash(): [0 for _ in range(a.utility_budget + 1)] for a in instance.agents}
            for a in self.utl:
                self.utl[a][-1] = 1

    def copy(self):
        copy_state = VectorState()
        copy_state.loc = copy.deepcopy(self.loc)
        copy_state.utl = copy.deepcopy(self.utl)
        copy_state.bers = copy.deepcopy(self.bers)
        copy_state.time_left = self.time_left
        return copy_state

    def hash(self):
        hash = tuple((ah, str(self.loc[ah])) for ah in self.loc), \
               tuple((vh, self.bers[vh].hash()) for vh in self.bers), \
               tuple((ah, tuple(self.utl[ah])) for ah in self.utl)
        return hash

    def calculate_vertex_estimate(self, vertex: Vertex):
        return self.bers[vertex.hash()].e()

    def bernoulli(self, vertex: Vertex):
        return self.bers[vertex.hash()].r

    def get_loc(self, a_hash: int):
        return self.loc[a_hash]

    def prob_u_at_least(self, a_hash: int, x):
        return 1 - sum(self.utl[a_hash][u] for u in range(x))

    def __str__(self):
        return str(self.loc)


'''def dict_to_np_arr(dict: Dict):
    if round(sum(dict.values()), 5) != 1:
        raise Exception("Sum of probabilities in distribution must be 1")
    arr = np.zeros((max([k for k in dict if dict[k] != 0]) + 1))
    for k in dict:
        if round(k) != k:
            raise Exception("Number of targets must be an integer!")
        if dict[k] != 0:
            arr[k] = dict[k]
    return arr
'''