import math
import random
import time

import VectorInstance
import instance_decoder
import State
from Timer import Timer

import EmpInstance
import Node

import InstanceManager
from GenQueue import PriorityQueue, RegularQueue, Stack, AstarQueue


def make_instance(def_inst, method='VEC'):
    if method == 'EMP':
        instance = EmpInstance.EmpInstance(def_inst)
    elif method == 'VEC':
        instance = VectorInstance.VectorInstance(def_inst)
    elif method == 'SEM':
        instance = EmpInstance.SemiEmpInstance(def_inst)
    else:
        raise Exception('Unrecognized type')
    return instance


def is_sorted_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


class Solver:
    def __init__(self, def_inst):
        self.visited_states = {}
        self.def_inst = def_inst
        self.dist_calculated = False
        self.all_pair_distances = {}
        self.num_of_states = None
        self.dup_det = True

        self.NUMBER_OF_SIMULATIONS = 999999
        self.DISCOUNT = 1
        self.timeout = 0
        self.num_of_logs = 20
        self.root = None
        self.instance = make_instance(def_inst)

        self.best_node = None
        self.best_value = 0
        self.map_reduced = False

        self.return_path = False
        self.timer = Timer()

    def restart(self):
        self.timer = Timer()
        self.timer.start('run')
        self.timer.start('log')
        self.root = Node.Node(None)
        self.root.state = self.instance.initial_state.copy()
        self.root.path = {a: [State.Action(self.root.state.get_loc(a), False)] for a in self.instance.agents_map}
        self.best_node = self.root
        self.best_value = self.instance.reward(self.root.state)
        self.num_of_states = 0

    def get_results(self):
        if self.return_path:
            return self.get_best_node().path
        log = self.timer.logs['run']
        results = []
        for t in log:
            reward = round(self.evaluate_path(log[t][0]), 3)

            '''reward_emp = round(self.evaluate_path(log[t][0], method='EMP'), 3)
            if reward_emp != reward:
                breakpoint()
            '''
            results.append((reward, log[t][1], round(t, 3)))
        return tuple(results)

    def get_best_node(self):
        if self.best_node is not None:
            return self.best_node
        best = self.root
        while not best.state.is_terminal() and len(best.children) != 0:
            best = best.highest_value_child()
        return best

    def log_if_needed(self, path=None):
        now = self.timer.now()
        if self.timer.duration_gt('log', self.timeout / self.num_of_logs, alt_now=now):
            self.force_log(path=path, alt_now=now)
            return True
        return False

    def force_log(self, alt_now=None, path=None):
        if alt_now is not None:
            now = alt_now
        else:
            now = self.timer.now()
        self.timer.restart('log', alt_now=now)
        self.timer.log(
            (self.get_best_node().path if path is None else path, self.num_of_states),
            thing='run', alt_now=now)

    def is_timeout(self):
        return self.timer.duration_gt('run', self.timeout)

    def base_upper_bound(self, state):
        possible_destinations_expectations = {}
        for agent in self.instance.agents:
            current_vertex = state.action[agent.hash()].loc
            for v in self.instance.map:
                if v.hash() == current_vertex or self.all_pair_distances[(v.hash(), current_vertex)] > (
                        agent.movement_budget - (self.instance.horizon - state.time_left)):
                    continue
                possible_destinations_expectations[v.hash()] = v.expectation()
        max_visits = sum([agent.movement_budget - (self.instance.horizon - state.time_left)
                          for agent in self.instance.agents])
        best_vertices = [k[0] for k in sorted(possible_destinations_expectations.items(), key=lambda item: item[1])][
                        0:max_visits - 1:]
        return sum([possible_destinations_expectations[v] for v in best_vertices])

    def calculate_all_pairs_distances_with_Seidel(self):
        self.all_pair_distances = InstanceManager.calculate_all_pairs_distances_with_Seidel(self.instance)
        self.dist_calculated = True

    def get_reachable_by_agents(self, state, agent=None):
        if agent is None:
            agents = self.instance.agents
        else:
            agents = [agent]
        reachable = []
        for agent in agents:
            reachable += [v for v in self.get_reachable_from(self.instance.map_map[state.loc[agent.hash()]],
                                                             self.instance.movement_left(state, agent)) if
                          v not in reachable]
        return reachable

    def get_reachable_from(self, ver, steps):
        reach = []
        for v in self.instance.map:
            d = self.all_pair_distances[(v.hash(), ver.hash())]
            if d <= steps:
                reach.append(v)
        return reach

    def get_sum_est_utility(self, state):
        estimated_utility_left = 0
        for agent in self.instance.agents:
            estimated_utility_left += self.get_est_utility(state, agent)
        return estimated_utility_left

    def get_est_utility(self, state, agent):
        return sum([state.utl[agent.hash()][u] * u for u in range(len(state.utl[agent.hash()]))])

    def get_reachable_exp_rewards(self, state):
        reachable_vertices = self.get_reachable_by_agents(state)
        reachable_exps = []
        for v in reachable_vertices:
            reachable_exps.append(v.expectation())
        return reachable_exps

    def get_reachable_bers(self, state):
        reachable_vertices = self.get_reachable_by_agents(state)
        reachable_bers = []
        for v in reachable_vertices:
            reachable_bers.append(state.bernoulli(v))
        return reachable_bers

    def upper_bound_base_plus_utility(self, state):
        reachable_bers = sorted(self.get_reachable_bers(state), reverse=True)
        est_utility = self.get_sum_est_utility(state)
        return sum(reachable_bers[:min(len(reachable_bers), math.ceil(est_utility))])

    def get_greedy_path(self, start, steps, state, visited):
        steps_left = steps
        ver = start
        path = []
        while steps_left > 0:
            reachable = [v for v in self.get_reachable_from(ver, steps_left) if v.hash() not in visited]
            best = max(reachable, key=lambda v: state.bers[v.hash()].e()) if len(reachable) != 0 else None
            if best is None or state.bers[best.hash()].e() == 0:
                return path
            steps_left -= max(self.all_pair_distances[ver.hash(), best.hash()], 1)
            path.append(best)
            visited.add(best.hash())
            ver = best
        return path

    def greedy_lower_bound(self, state):
        agents = sorted(self.instance.agents, key=lambda a: self.instance.movement_left(state, a))
        visited_vertices = set()
        reward = 0
        for agent in agents:
            steps = self.instance.movement_left(state, agent)
            path = self.get_greedy_path(self.instance.map_map[state.loc[agent.hash()]],
                                        steps, state, visited_vertices)
            for u in range(1, agent.utility_budget + 1):
                reward += state.utl[agent.hash()][u] * sum(state.bernoulli(v) for v in path[0: min(u, len(path))])
        return reward

    def map_reduce(self):
        InstanceManager.map_reduce(self.instance)
        self.map_reduced = True

    def bfs(self):
        return self.branch_and_bound()

    def branch_and_bound(self, upper_bound=None, lower_bound=None, is_greedy=False, depth_first=False, astar=False):
        show_time = False
        # self.dup_det = False
        if upper_bound is not None or lower_bound is not None:
            self.calculate_all_pairs_distances_with_Seidel()
        self.restart()
        if show_time:
            self.timer.start("init")
        if is_greedy:
            que = PriorityQueue(self.root)
        elif astar:
            que = AstarQueue(self.root)
        elif depth_first:
            que = Stack(self.root)
        else:
            que = RegularQueue(self.root)
        if upper_bound is not None:
            lowest_bound = self.root
            lowest_bound.low = lower_bound(lowest_bound.state) if lower_bound is not None else 0
        if show_time:
            self.timer.end('init')
        while not que.is_empty():
            if self.is_timeout():
                if show_time:
                    print(str(self.timer))
                if not self.return_path:
                    self.force_log()
                return self.get_results()
            if not self.return_path:
                self.log_if_needed()
            node = que.pop()

            if show_time:
                self.timer.end_from_last_end('pop')
            if not node.state.is_terminal():
                node.expand(self.instance)
                self.num_of_states += len(node.children)

                if show_time:
                    self.timer.end_from_last_end('expand')
                for child in node.children:

                    if self.dup_det:
                        if self.is_duplicate(child.state):
                            continue

                    if show_time:
                        self.timer.end_from_last_end("dup det")

                    child.value = self.instance.reward(child.state)

                    if child.value > self.best_value:
                        self.best_node = child
                        self.best_value = child.value

                    if upper_bound is not None:
                        child.high = upper_bound(child.state)
                        if child.high + child.value < lowest_bound.value + lowest_bound.low:
                            continue

                        child.low = lower_bound(child.state) if lower_bound is not None else 0
                        if child.value + child.low > lowest_bound.value + lowest_bound.low:
                            lowest_bound = child
                    if show_time:
                        self.timer.end_from_last_end('value games')
                    que.push(child)
                    if show_time:
                        self.timer.end_from_last_end('push')
        if not self.return_path:
            self.force_log()
        return self.get_results()

    def value_plus_upper_bound(self, state):
        return self.instance.reward(state) + self.upper_bound_base_plus_utility(state)

    def is_duplicate(self, state):
        hash = state.hash()
        if hash not in self.visited_states:
            self.visited_states[hash] = state.time_left
            return False
        if state.time_left > self.visited_states[hash]:
            self.visited_states[hash] = state.time_left
            return False

        return True

    def emp_mcts(self):
        self.instance = make_instance(self.def_inst, method='EMP')
        return self.mcts('EMP')

    def vector_mcts(self):
        return self.mcts('VEC')

    def semi_emp_mcts(self):
        self.instance = make_instance(self.def_inst, method='SEM')
        return self.mcts('SEM')

    def mcts(self, method):
        self.restart()
        best_path = None
        self.best_node = None
        for t in range(self.NUMBER_OF_SIMULATIONS):
            if self.is_timeout():
                if not self.return_path:
                    self.force_log(path=best_path)
                return self.get_results()
            if not self.return_path:
                self.log_if_needed(best_path)
            node = self.root
            # selection

            while node.children:
                node.times_visited += 1
                node = node.highest_uct_child(t)

            # expansion
            if not node.state.is_terminal():
                if len(node.children) > 0:
                    breakpoint()
                children = [self.instance.make_action(action, node.state) for action in
                            self.instance.actions(node.state, node.path)]
                random.shuffle(children)
                node.expand(self.instance)
                self.num_of_states += len(node.children)
                node.times_visited += 1
                node = node.children[0]

            # simulation
            node.times_visited += 1
            rollout_state = node.state.copy()
            if method == 'VEC':
                path = node.path

            while not rollout_state.is_terminal():
                action = random.choice(self.instance.actions(rollout_state, node.path))
                if method == 'VEC':
                    for a in path:
                        path[a].append(action[a])
                rollout_state = self.instance.make_action(action, rollout_state)
            rollout_reward = self.instance.reward(rollout_state)

            # Deterministic approach allows us to memorize the best path
            if method == 'VEC' and rollout_reward > self.best_value:
                self.best_value = rollout_reward
                best_path = path

            discounted_reward = rollout_reward * pow(self.DISCOUNT, node.depth)

            # backpropagation
            while True:
                if (not node.all_children_visited()) or node.state.is_terminal():
                    avg_of_node = (node.value * (node.times_visited - 1) + discounted_reward) / node.times_visited
                    node.value = avg_of_node
                    discounted_reward = avg_of_node
                else:
                    if node.value < discounted_reward:
                        node.value = discounted_reward
                    else:
                        break
                if node is self.root:
                    break
                node = node.parent
                discounted_reward /= self.DISCOUNT

            # gathering data

        # root.get_tree()
        # returning
        if not self.return_path:
            self.force_log(path=best_path)
        return self.get_results()

    def evaluate_path(self, path, method='VEC'):
        self.instance = make_instance(self.def_inst, method)
        if path is None:
            return 0
        state = self.instance.initial_state.copy()
        for t in range(1, len(list(path.values())[0])):
            action = {a: State.Action(path[a][t].loc, path[a][t].dropoff) for a in path}
            state = self.instance.make_action(action, state)
        reward = self.instance.reward(state) if method == 'VEC' \
            else self.instance.average_of_sims(state, 10000)
        return reward


if __name__ == "__main__":
    pass
