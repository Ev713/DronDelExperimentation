import concurrent.futures
import multiprocessing
import sys
import time
import tracemalloc
from multiprocessing import Process

import pandas as pd
import psutil

import InstanceManager
import Solver
import instance_decoder


def write_data(r, name, ):
    fin_res, t, res, states, inst, algo = r[0], r[1], r[2], r[3], r[4], r[5]
    df = pd.DataFrame({'inst_name': [str(inst.name)], 'num_agents': [len(inst.agents)], 'map_size': [len(inst.map)],
                       'source': [str(inst.source)], 'type': [str(inst.type)], 'horizon': [int(inst.horizon)],
                       'algo': [str(algo)],
                       'final_result': [float(fin_res)], 'time': [float(t)] if t != '-' else '-', 'states': int(states),
                       'result': [str(res)]})
    print({'inst_name': inst.name, 'num_agents': len(inst.agents), 'map_size': len(inst.map),
           'source': inst.source, 'type': inst.type, 'horizon': inst.horizon, 'algo': algo,
           'final_result': fin_res, 'time': t, 'states': states, })
    try:
        df.to_csv('data/' + name + '.csv', mode='a', index=False,
                  header=False)
    except:
        print("Save failed, create \'data\' folder to save.")


def run_solver(inst, algo, timeout=1800, default='-', return_path=False):
    print("Start " + inst.name, algo)
    solver = Solver.Solver(inst)
    solver.dup_det = True
    solver.timeout = timeout
    solver.return_path = return_path
    results = None
    if algo == 'MCTS_E':
        results = solver.emp_mcts()
    if algo == 'MCTS_V':
        results = solver.vector_mcts()
    if algo == 'MCTS_S':
        results = solver.semi_emp_mcts()
    if algo == 'BFS':
        solver.type = 'U1S'
        results = solver.bfs()
    if algo == 'BNBL':
        solver.type = 'U1S'
        results = solver.branch_and_bound(solver.upper_bound_base_plus_utility,
                                          solver.greedy_lower_bound)
    if algo == 'BNB':
        solver.type = 'U1S'
        results = solver.branch_and_bound(solver.upper_bound_base_plus_utility)
    if algo == 'GBNB':
        solver.type = 'U1S'
        results = solver.branch_and_bound(solver.upper_bound_base_plus_utility,
                                          solver.greedy_lower_bound, is_greedy=True)
    if algo == 'ASTAR':
        solver.type = 'U1S'
        results = solver.branch_and_bound(solver.upper_bound_base_plus_utility,
                                          solver.greedy_lower_bound, astar=True)
    if algo == 'DFS':
        solver.type = 'U1S'
        results = solver.branch_and_bound(depth_first=True)
    if return_path:
        return results
    fin_res = results[-1][0] if len(results) > 0 else default
    fin_time = results[-1][-1] if len(results) > 0 and results[-1][-1] < timeout else default
    fin_states = results[-1][1] if len(results) > 0 else default
    return fin_res, fin_time, results, fin_states, inst, algo


def single_run():
    timeout = 20
    decoder = instance_decoder.Decoder()
    decoder.decode(file_path='maps')
    inst = decoder.instances[0]
    name = 'scratch'
    # Inst_visualizer.vis3(inst, name)
    algo = 'GBNB'
    solve(inst, algo, timeout, name)


def solve(*args):
    write_data(run_solver(args[0], args[1], args[2]), args[3])


def multi_run():
    algos = [
        'MCTS_E',
        'MCTS_V',
        #'MCTS_S',
        'BFS',
        #'BNBL',
        'BNB',
        'GBNB',
        #'ASTAR',
        # 'DFS'
    ]
    name = 'apr_11'
    timeout = 20
    start = time.perf_counter()
    decoder = instance_decoder.Decoder()
    decoder.decode(file_path='maps')
    instances = decoder.instances
    runs_left = len(instances) * len(algos)
    max_workers = 2  # round(multiprocessing.cpu_count() * 0.10)
    print(f"Starting multi-run. \nTimeout: {timeout}\n"
          f"Algorithms: {algos}\nMax workers: {max_workers}\n"
          f"Instances: {len(instances)}")

    processes = []
    killed = 0
    for inst in instances:
        for algo in algos:
            p = multiprocessing.Process(target=solve, args=(inst, algo, timeout, name))
            p.start()
            processes.append(p)
            print(f"number of processes: {len(processes)}")
            last_start = time.perf_counter()
            ram = psutil.virtual_memory()[2]
            if len(processes) >= max_workers or ram > 50:
                while (all([p.is_alive() for p in processes]) and (len(processes) >= max_workers
                                                                   or runs_left == len(processes))) or ram > 50:
                    ram = psutil.virtual_memory()[2]
                    if ram > 75:
                        processes[0].kill()
                        killed += 1
                    time_passed = round(time.perf_counter() - last_start)
                    if time_passed != 0:
                        print(f"Waiting for a process to finish for {time_passed} seconds."
                              f" Expected time: {timeout - time_passed}.\n"
                              f"Memory used: {ram}%")
                    time.sleep(10)

                for p in processes:
                    if not p.is_alive():
                        processes.remove(processes[0])
                        runs_left -= 1
                        print(f"Process removed. New process may start. {runs_left} left")
                print(f"number of processes: {len(processes)}")

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')
    print(f"Processes killed: {killed}")


if __name__ == "__main__":
    decoder = instance_decoder.Decoder()

    decoder.decode(file_path='json_maps', from_json=True)
    instances = decoder.instances
    for inst in instances:
        inst0 = inst


