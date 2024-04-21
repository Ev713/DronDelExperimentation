import Agent
import State
import Vertex
import Instance
from typing import Dict, List, Optional


class EmpInstance(Instance.Instance):
    def __init__(self, instance=None):
        super().__init__(instance.name, instance.map, instance.agents, instance.horizon)
        self.map, self.map_map = instance.make_special_map_and_map_map(Vertex.EmpVertex)
        self.agents, self.agents_map = instance.make_agents_and_agents_map( Agent.DetAgent)
        self.horizon = instance.horizon
        self.initial_state = State.EmpState(instance)
        self.dropoffs = instance.dropoffs

    def regenerate_instance(self):
        self.generate_rewards()
        for a in self.agents:
            a.current_utility_budget = a.utility_budget

    def generate_rewards(self):
        for v in self.map:
            v.generate_reward()

    def make_action(self, action: Dict[int, State.Action], state: State.State):
        new_state = state.copy()
        new_state.time_left -= 1
        time = self.horizon - new_state.time_left
        for a_hash in action.keys():
            new_state.path[a_hash][time] = action[a_hash]
        return new_state

    def reward(self, state: State.EmpState):
        return self.average_of_sims(state, 1)

    def average_of_sims(self, state: State.EmpState, num_of_sims: int):
        tot_reward = 0
        for _ in range(num_of_sims):
            self.regenerate_instance()
            round_reward = 0
            for t in range(1, self.horizon + 1):
                for a in self.agents:
                    if a.movement_budget < t or \
                            a.current_utility_budget == 0 or \
                            state.path[a.hash()][t] is None:
                        continue
                    a_loc = self.map_map[state.path[a.hash()][t].loc]
                    if a_loc is None or a_loc.is_empty or not state.path[a.hash()][t].dropoff:
                        continue
                    a_loc.is_empty = True
                    round_reward += a_loc.reward
                    a.current_utility_budget -= 1
            tot_reward += round_reward
        return tot_reward / num_of_sims


class SemiEmpInstance(EmpInstance):
    def __init__(self, instance=None):
        super().__init__(instance)

    def generate_rewards(self):
        for v in self.map:
            v.generate_semi_emp_reward()
