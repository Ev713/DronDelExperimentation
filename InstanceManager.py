import itertools
import json

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



def to_json(inst, filepath=''):
    # Create a dictionary to hold the data
    data = {
        'name': inst.name,
        'horizon': inst.horizon,
        'source': inst.source,
        'agents': [
            {
                'id': a.id,
                'location_hash': a.loc,
                'movement_budget': a.movement_budget,
                'utility_budget': a.utility_budget
            }
            for a in inst.agents
        ],
        'map': [
            {
                'vertex_hash': v.hash(),
                'neighbours': v.neighbours,
                'distribution': {
                    r: v.distribution.get(r, 0) for r in range(max(list(v.distribution.keys())) + 1)
                }
            }
            for v in inst.map
        ]
    }

    # Define the path to save the JSON file
    json_filepath = f'{filepath}/{inst.name}.json'

    # Write the JSON data to a file
    with open(json_filepath, 'w') as file:
        json.dump(data, file, indent=4)


def json_to_inst(filepath):
    # Read the JSON data from the file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Extract the basic information
    name = data['name']
    horizon = data['horizon']
    source = data['source']

    # Create a map of vertex hash to vertex objects
    vertex_map = {}
    for vertex_data in data['map']:
        vertex = Vertex.Vertex(vertex_data['vertex_hash'])
        vertex_map[vertex.hash()] = vertex

    # Set up the vertex neighbours
    for vertex_data in data['map']:
        vertex = vertex_map[vertex_data['vertex_hash']]
        for neighbour_hash in vertex_data['neighbours']:
            vertex.neighbours.append(neighbour_hash)
        vertex.distribution = {int(r): vertex_data['distribution'][r] for r in vertex_data['distribution']}

    # Create agents
    agents = []
    for agent_data in data['agents']:
        agent = Agent.Agent(None, None, None, None)
        agent.id = agent_data['id']
        # Locate the vertex by hash
        agent.loc = vertex_map[agent_data['location_hash']]
        agent.movement_budget = agent_data['movement_budget']
        agent.utility_budget = agent_data['utility_budget']
        agents.append(agent)

    # Create and return the instance
    return Instance.Instance(name, list(vertex_map.values()), agents, horizon, source)


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
