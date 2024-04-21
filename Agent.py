import numpy as np

import Vertex


class Agent:
    def __init__(self, id: int, loc: int, movement_budget: int, utility_budget: int):
        self.id: int = id
        self.loc: int = loc  # vertex location
        self.movement_budget: int = movement_budget  # int
        self.utility_budget: int = utility_budget  # int

    def hash(self):
        if self.id == -1:
            raise Exception("-1 is an unusable hash number.")
        return self.id

    def __str__(self):
        return "a" + str(self.id)


class StochAgent(Agent):
    def __init__(self, number: int, loc: int, movement_budget: int, utility_budget: int):
        super().__init__(number, loc, movement_budget, utility_budget)


class DetAgent(Agent):
    def __init__(self, number: int, loc: int, movement_budget: int, utility_budget: int):
        super().__init__(number, loc, movement_budget, utility_budget)
        self.current_utility_budget = utility_budget
