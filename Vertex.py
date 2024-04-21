import random
import typing

class Vertex:
    def __init__(self, id: int):
        self.id = id
        self.neighbours: typing.List[int] = []  # list of Vertices
        self.distribution: typing.Dict[float, float] = {}  # reward: probability dictionary

    def hash(self):
        if self.id == -1:
            raise Exception("-1 is an unusable hash number.")
        return self.id

    def __str__(self):
        return "v" + str(self.id)

    def expectation(self):
        return sum([r * self.distribution[r] for r in self.distribution])

    def p(self):
        return 1 - self.distribution[0]

    def q(self):
        return self.distribution[0]

    def bernoulli(self):
        return 0 if self.p() == 0 else sum([r * self.distribution[r] for r in self.distribution]) / self.p()


class EmpVertex(Vertex):
    def __init__(self, id):
        super().__init__(id)
        self.is_empty: bool = False
        self.reward: float = 0

    def generate_reward(self):
        p = random.random()
        sum_p = 0
        for r in self.distribution:
            sum_p += self.distribution[r]
            if sum_p > p:
                self.reward = r
                break
        self.is_empty = (self.reward == 0)

    def generate_semi_emp_reward(self):
        if random.random() > self.q():
            self.reward = self.bernoulli()
        self.is_empty = (self.reward == 0)

    def __str__(self):
        return "det_v" + str(self.id) + " " + str(self.reward) + " " + str(self.is_empty)


class Stoch_Vertex(Vertex):
    def __init__(self, id):
        super().__init__(id)
