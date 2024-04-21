import statistics

import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # Use Tkinter backend, you can replace 'TkAgg' with another backend that works for you

import matplotlib.pyplot as plt
import numpy as np
import math

# (inst.name, solver_type, algo, flybys)

renames = {
    'MT': 'Mountain-Top',
    'SC': 'Sanity-Check',
    'AG': 'Anti-Greedy',
    'FR': 'Full-Random',
    'FL': 'Flower',

    'MCTS_S': 'UCT',
    'MCTS_V': f'UCT$^D$',
    'BFS': 'BFS',
    'ASTAR': 'BNB',
    'BNB': 'BnB Breadth first, lower = 0',
    'BNBL': 'BnB Breadth first',
    'MCTS_E': 'UCT with full distribs',
    'GBNB': 'Greedy BnB'

}


class Run:
    def __init__(self):
        self.inst_name = None
        self.solver_type = None
        self.algo = None
        self.flybys = None
        self.results = None
        self.time = None
        self.fin_res = None
        self.states = 0
        self.map_type = None
        # self.is_real = False
        self.size = 0

    def ag_is_success(self):
        return self.fin_res == (self.size - 1) / 2 + 1

    def sc_is_success(self):
        return self.fin_res == math.sqrt(self.size - 1) * 2

    def is_timeout(self):
        return self.time == -1

    def inst_name_extract_type(self):
        if self.inst_name[0] == '_':
            ar_id = self.inst_name.find('AR')
            self.map_type = self.inst_name[ar_id - 2] + self.inst_name[ar_id - 1]
            self.is_real = True
        else:
            self.map_type = self.inst_name[-2] + self.inst_name[-1]
            self.is_real = False

    def copy(self):
        cp = Run()
        cp.inst_name = self.inst_name
        cp.time = self.time
        cp.solver_type = self.solver_type
        cp.algo = self.algo
        cp.flybys = self.flybys
        cp.results = self.results
        cp.fin_res = self.fin_res
        cp.map_type = self.map_type
        cp.is_real = self.is_real


class Analyzer:
    def __init__(self, filepath):
        self.data_for_graphs = None
        self.acc = None
        self.instaces = None
        self.timeout = None
        self.allowed_types = None
        self.algos = None
        self.data_for_tables = None
        self.file_path = filepath
        self.df = pd.read_csv(self.file_path, header=None, on_bad_lines='skip', engine='python')
        self.runs = []
        self.instances = {}

    def create_runs(self):
        for i in range(len(self.df)):
            row = self.df.loc[i, :].to_list()
            run = Run()
            run.inst_name = row[0].strip("\"")
            run.num_of_agents = row[1]
            run.size = row[2]
            run.source = row[3]
            run.type = row[4]
            run.horizon = row[5]
            run.algo = row[6]
            run.fin_res = row[7]
            run.time = float(row[8]) if row[8] != '-' else -1
            run.states = row[9]
            run.results = [[float(number.strip(",)")) for number in pair.split(", ")] for pair in
                           row[10].strip("()").strip().split("), (")]

            self.runs.append(run)

    def get_sat_graph(self):
        fin_ress = {algo: [] for algo in self.algos}
        sizes = {algo: [] for algo in self.algos}
        states = {algo: [] for algo in self.algos}
        default = 'BFS'
        num_of_types = {type: 0 for type in self.allowed_types}
        def_time = self.timeout  # max([run.results[-1][2] for run in analyzer.runs])  # instance_runs[default].results[-1][2]
        for inst_name in self.instances:

            instance_runs = self.instances[inst_name]

            # all_algos = True
            # for algo in algos:
            #    if algo not in instance_runs:
            #        all_algos = False

            # if not all_algos:
            #    continue

            if instance_runs[list(instance_runs.keys())[0]].type not in self.allowed_types:
                continue

            # if instance_runs['BFS'].source == 'X' or instance_runs['BFS'].type != 'MT':
            #    continue

            max_result = max([instance_runs[algo].results[-1][0] for algo in self.algos if algo in instance_runs])
            def_states = 1  # max([instance_runs[algo].results[-1][1] for algo in self.algos if algo in instance_runs])

            # for instance in instances:
            #    for algo in algos:
            #        if algo in instances[instance]:
            #            num_of_types[instances[instance][algo].type] += 1
            #            break

            for algo in self.algos:
                if algo not in instance_runs:
                    continue
                run = instance_runs[algo]
                size = run.size
                fin_res = run.results[-1][0]

                sizes[algo].append(size)
                fin_ress[algo].append(fin_res)
                states[algo].append(run.states)

                for pair in run.results:
                    pair[0] = 0 if max_result == 0 else \
                        round(pair[0] / max_result, self.acc)

                    if pair[0] > 1:
                        breakpoint()
                    pair[1] = pair[1]
                    pair[2] = round(pair[2])
                if algo not in self.data_for_graphs:
                    self.data_for_graphs[algo] = []
                self.data_for_graphs[algo].append(run)
                if algo not in self.data_for_tables:
                    self.data_for_tables[algo] = []
                self.data_for_tables[algo].append(run.states / def_states)

        for t in num_of_types:
            print(t, num_of_types[t])

        relative_to_states = False

        graphs = {algo: [[], []] for algo in self.data_for_graphs}
        for algo in self.data_for_graphs:
            runs = self.data_for_graphs[algo]
            runs_complete_results = []
            for run in runs:
                run_complete_data = {0: 0}
                for t in range(def_time):
                    for r in run.results:
                        if r[2] == t:
                            run_complete_data[t] = r[0]  # 0 for result 1 for states
                    if t not in run_complete_data:
                        run_complete_data[t] = run_complete_data[t - 1]
                runs_complete_results.append(run_complete_data)
            for t in range(0, def_time, int(def_time / 20)):
                results = []
                for run in runs_complete_results:
                    results.append(run[t])
                avg_result = statistics.mean(results)
                graphs[algo][0].append(t)
                graphs[algo][1].append(avg_result)

        plt.figure(figsize=(6.4, 5))

        markers = ['>', '+', '.', ',', 'o', 'v', 'x', 'X', 'D', '|']

        for i in range(len(self.algos)):
            plt.plot(graphs[self.algos[i]][0], graphs[self.algos[i]][1], markers[i % 10], linestyle='-')

        if not relative_to_states:
            plt.xlabel("Time")
        else:
            plt.xlabel("States")
        plt.ylabel("Result (relative to best)")
        plt.title(self.get_title() + " Anytime")
        plt.ylim(bottom=0)
        plt.xlim(self.timeout / 20, self.timeout)


        #plt.legend([renames[algo] for algo in self.algos])
        plt.tight_layout()
        print("States (relative to best states):")
        for algo in self.data_for_tables:
            print(algo + ": " + str(round(statistics.mean(self.data_for_tables[algo]), 3)))
        plt.savefig("data/17apr/Images/" + self.get_title().replace(" ", '_') +
                    "sat" + ("_SM" if len(self.algos) > 4 else "") + ".jpg")
        plt.show()

    def get_opt_graph(self):

        if 'MCTS_S' in self.algos:
            self.algos.remove('MCTS_S')
        if 'MCTS_E' in self.algos:
            self.algos.remove('MCTS_E')
        if 'MCTS_V' in self.algos:
            self.algos.remove('MCTS_V')
        opt_results = {}
        for inst_name in self.instances:
            inst = self.instances[inst_name]
            for algo in inst:
                if -1 < inst[algo].time < self.timeout * 0.9:
                    opt_results[inst_name] = inst[algo].fin_res
                    break

        self.data_for_graphs = {algo: [[], []] for algo in self.algos}

        for algo in self.algos:
            for t in range(0, self.timeout, int(self.timeout / 20)):
                num_of_succ = 0
                for run in self.runs:
                    if run.inst_name not in opt_results:
                        continue
                    if run.algo == algo and -1 < run.time <= t:
                        if run.fin_res != opt_results[run.inst_name]:
                            raise Exception("different final solutions")
                        if run.type in self.allowed_types:
                            num_of_succ += 1
                self.data_for_graphs[algo][0].append(t)
                self.data_for_graphs[algo][1].append(num_of_succ)

        markers = ['>', '+', '.', ',', 'o', 'v', 'x', 'X', 'D', '|']

        self.algos = sorted(['MCTS_V',
        'MCTS_S',
        'ASTAR',
        'BFS',])

        inviss = []
        plt.figure(figsize=(6.4, 6.6))
        for i in range(len(self.algos)):
            if not 'MCTS' in self.algos[i]:
                plt.plot(self.data_for_graphs[self.algos[i]][0],
                         self.data_for_graphs[self.algos[i]][1], markers[i % 10], linestyle='-')
            else:
                inviss.append(plt.plot([1],[1], markers[i % 10], linestyle='-'))

        plt.legend([renames[algo] for algo in self.algos], ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3))
        plt.xlabel("Time")
        plt.ylabel("Number of Solved")
        plt.xlim(0, self.timeout)

        plt.title(self.get_title() + " Optimal")
        plt.ylim(bottom=0)
        plt.tight_layout()

        plt.savefig("data/17apr/Images/" + self.get_title().replace(" ", '_') + "opt" + (
            "_SM" if len(self.algos) > 4 else "") + ".jpg")

        plt.show()

    def get_title(self):
        used_types = {t for t in self.allowed_types if t not in ('AG05', 'AG01', 'AG001')}
        if 'AG05' in self.allowed_types:
            used_types.add('AG')
        if len(used_types) >= 4:
            return "Overall"
        types = ""
        for t in used_types:
            types += f"{renames[t]}   "
        return types


def main():
    filepath = "data/paper_opt.csv"
    analyzer = Analyzer(filepath)
    analyzer.acc = 2
    analyzer.timeout = 900
    analyzer.allowed_types = (
        'FR',
        'MT',
        'SC',
        'AG01', 'AG001', 'AG05',
    )
    allowed_algos = (
        'MCTS_V',
        'MCTS_S',
        'ASTAR',
        'BFS',

        # 'BNBL',
        # 'BNB',
        # 'MCTS_E',
        # 'GBNB'
    )

    analyzer.create_runs()
    analyzer.instances = {}
    analyzer.data_for_graphs = {}
    analyzer.data_for_tables = {}
    analyzer.algos = set()
    counter = 0

    for run in analyzer.runs:
        if run.algo in allowed_algos:
            analyzer.algos.add(run.algo)
        else:
            continue
        max_result = 0
        for r in run.results:
            if r[0] >= max_result:
                max_result = r[0]
            else:
                counter += 1
    analyzer.algos = sorted(list(analyzer.algos))

    for run in analyzer.runs:
        if run.inst_name not in analyzer.instances:
            analyzer.instances[run.inst_name] = {}
        analyzer.instances[run.inst_name][run.algo] = run

    font = {'family': 'normal',
            'size': 25}

    matplotlib.rc('font', **font)

    analyzer.get_opt_graph()

    # algo = 'BNBL'
    # plt.scatter(sizes[algo], fin_ress[algo])
    # plt.title('BNB')
    # plt.xlabel("Size")
    # plt.ylabel("Reward")

    # , sizes['BNB'], fin_ress['BNB'], sizes['BNBL'], fin_ress['BNBL'],
    #            sizes['MCTS_S'], fin_ress['MCTS_S'], sizes['MCTS_D'], fin_ress['MCTS_D'])

    # plt.savefig("data/Images/server_full_data.png")


if __name__ == '__main__':
    main()
