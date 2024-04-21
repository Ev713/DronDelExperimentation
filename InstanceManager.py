import itertools

import numpy

import Vertex
import Instance
import Agent
from numpy import array as matrix


def apd(A, n: int):
    """Compute the shortest-paths lengths."""
    if all(A[i][j] for i in range(n) for j in range(n) if i != j):
        return A
    Z = numpy.matmul(A, A)
    B = matrix([
        [1 if i != j and (A[i][j] == 1 or Z[i][j] > 0) else 0 for j in range(n)]
        for i in range(n)])
    T = apd(B, n)
    X = numpy.matmul(T, A)
    degree = [sum(A[i][j] for j in range(n)) for i in range(n)]
    D = matrix([
        [2 * T[i][j] if X[i][j] >= T[i][j] * degree[j] else 2 * T[i][j] - 1 for j in range(n)]
        for i in range(n)])
    return D


def to_string(inst, filepath=''):
    file = open(filepath + '/' + inst.name + ".txt", mode='w')
    file.write(str(inst.name) + '\n')
    file.write(str(inst.horizon) + '\n')
    file.write(inst.source + '\n')
    for a in inst.agents:
        file.write('A' + '\n')
        file.write(str(a.id) + '\n')
        file.write(str(a.loc.hash()) + '\n')
        file.write(str(a.movement_budget) + '\n')
        file.write(str(a.utility_budget) + '\n')
    for v in inst.map:
        file.write('V' + '\n')
        file.write(str(v.hash()) + '\n')
        file.write('N' + '\n')
        for n in v.neighbours:
            file.write(str(n.hash()) + '\n')
        file.write('D' + '\n')
        for r in range(max(list(v.distribution.keys())) + 1):
            if r not in v.distribution:
                file.write('0' + '\n')
            else:
                file.write(str(v.distribution[r]) + '\n')


def map_reduce(inst):
    want_to_print = False
    if not inst.map_is_connected():
        return
    if want_to_print:
        print("Gathering vertices that may not be empty or are initial locations for agents. ")
    calculate_all_pairs_distances_with_Seidel(inst)
    essential_vertices = []
    init_locs = []
    for a in inst.agents:
        init_locs.append(a.loc)
    for v in inst.map:
        if ((v.distribution[0] < 1) or (v in init_locs)) and v not in essential_vertices:
            essential_vertices.append(v)
    if want_to_print:
        print("Essential vertices: ", len(essential_vertices))
        print("Determining vertices that may be used in an optimal run.")

    is_used = set()
    for start in essential_vertices:
        if want_to_print:
            print("Start ", start.id)
        for end in essential_vertices:
            if want_to_print:
                print("End ", end.id)
            if end == start:
                continue
            queue = [(start, [])]
            checked = []
            while True:
                v, path_to_v = queue.pop()
                checked.append(v)
                if v == end:
                    for t in path_to_v:
                        if t not in is_used:
                            is_used.add(t)
                    break
                else:
                    for t in v.neighbours:
                        if t not in checked:
                            queue.insert(0, (t, path_to_v + [v]))
                            checked.append(t)

    print("Creating new map that contains only \"useful\" vertices")
    new_map = []
    for v in inst.map:
        if (v in is_used) or (v in essential_vertices):
            new_neighbours = []
            for j in v.neighbours:
                if (j in is_used) or (j in essential_vertices):
                    new_neighbours.append(j)
            v.neighbours = new_neighbours
            new_map.append(v)
    inst.map = new_map
    for a in inst.agents:
        a.loc = new_map[0]
    print("Done")


def calculate_all_pairs_distances_with_Seidel(inst):
    n = len(inst.map)
    A = matrix([[1 if (inst.map[j].id in inst.map[i].neighbours) else 0 for i in range(n)] for j in range(n)])
    assert numpy.sum(A) > 0
    D = apd(A, n)
    return {(inst.map[i].hash(), inst.map[j].hash()): D[i][j] for i in range(n) for j in range(n)}


def to_inst(filepath):
    file = open(filepath, mode='r')
    line_it = iter(file)
    name = next(line_it).strip()
    horizon = int(next(line_it).strip())
    source = next(line_it).strip()
    agents = []
    map = []
    map_map = {}
    neighbours_hashes = {}
    EOF = False
    while next(line_it).strip() == 'A':
        agent = Agent.Agent(-1, -1, -1, -1)
        agent.id = int(next(line_it).strip())
        agent.loc = int(next(line_it).strip())
        agent.movement_budget = int(next(line_it).strip())
        agent.utility_budget = int(next(line_it).strip())
        agents.append(agent)
    while True:
        vertex = Vertex.Vertex(int(next(line_it).strip()))
        neighbours_hashes[vertex.hash()] = []
        if not next(line_it).strip() == 'N':
            raise Exception('Instance encoded incorrectly!')
        while True:
            n = next(line_it).strip()
            if n == 'D':
                break
            vertex.neighbours.append(int(n))
        for r in itertools.count(start=0):
            next_line = next(line_it, 'EOF').strip()
            if next_line == 'V':
                break
            if next_line == 'EOF':
                EOF = True
                break
            vertex.distribution[r] = float(next_line)
        map.append(vertex)
        map_map[vertex.hash()] = vertex
        if EOF:
            return Instance.Instance(name, map, agents, horizon, source)


def filter_unconnected(inst):
    neighbours = {v.hash(): [n.hash() for n in v.neighbours] for v in inst.map}
    connected_component_size = {}
    connected = {}
    for vertex in inst.map:
        connected[vertex.hash()] = [vertex.hash()]
        no_more_connected_vertices = False
        while not no_more_connected_vertices:
            no_more_connected_vertices = True
            for v in connected[vertex.hash()]:
                for n in neighbours[v]:
                    if n not in connected[vertex.hash()]:
                        connected[vertex.hash()].append(n)
                        no_more_connected_vertices = False

        connected_component_size[vertex.hash()] = len(connected[vertex.hash()])
    vertex_in_biggest_connected = [k[0] for k in sorted(connected_component_size.items(), key=lambda item: item[1])][-1]
    new_map = []
    for v in inst.map:
        if v.hash() in connected[vertex_in_biggest_connected]:
            new_map.append(v)
    inst.map = new_map
