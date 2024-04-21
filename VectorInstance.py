from abc import ABC

import numpy as np

import State
import Instance
import Vertex
import Agent
import Instance
import copy
from typing import List, Dict, Optional


class VectorInstance(Instance.Instance):
    def __init__(self, instance: Instance):
        super().__init__(instance.name, instance.map, instance.agents, instance.horizon)
        self.map, self.map_map = instance.make_special_map_and_map_map(Vertex.Stoch_Vertex)
        self.agents, self.agents_map = instance.make_agents_and_agents_map(Agent.StochAgent)
        self.horizon = instance.horizon
        self.initial_state = State.VectorState(instance)
        self.initial_state.time_left = self.horizon
        self.dropoffs = instance.dropoffs

    def make_action(self, action: Dict[int, State.Action], state: State):
        new_state = state.copy()
        new_state.time_left -= 1
        for a_hash in self.agents_map:
            new_state.loc[a_hash] = action[a_hash].loc
            if (not action[a_hash].dropoff) or \
                    self.agents_map[a_hash].movement_budget < self.horizon - new_state.time_left:
                continue
            ber = new_state.bers[action[a_hash].loc]
            for u in range(1, len(new_state.utl[a_hash])):
                new_state.utl[a_hash][u - 1] += new_state.utl[a_hash][u] * ber.p
                new_state.utl[a_hash][u] *= ber.q()
            if round(sum(new_state.utl[a_hash]), 5) != 1:
                raise Exception('New distr sum not 1')
            new_state.bers[action[a_hash].loc].p *= state.utl[a_hash][0]
        return new_state

    def movement_left(self, state: State.State, agent: Agent.Agent):
        return agent.movement_budget - self.get_time(state)

    def reward(self, state: State.VectorState):
        if state.reward is not None:
            return state.reward
        state.reward = 0
        for v_hash in self.map_map:
            state.reward += self.map_map[v_hash].expectation() - state.bers[v_hash].e()
        return state.reward
