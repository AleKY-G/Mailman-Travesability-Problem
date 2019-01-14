########################################################################################################################
########################################################################################################################
#Analytics              _       _   _          #
#     /\               | |     | | (_)         #
#    /  \   _ __   __ _| |_   _| |_ _  ___ ___ #
#   / /\ \ | '_ \ / _` | | | | | __| |/ __/ __|#
#  / ____ \| | | | (_| | | |_| | |_| | (__\__ \#
# /_/    \_\_| |_|\__,_|_|\__, |\__|_|\___|___/#
#                          __/ |               #
#                         |___/                #
################################################
import sys
inf = float('inf')

def Weight(graph_input, dijkstra_true=False, start=0, end=0):
    graph = []
    graphcopy = []
    for i in range(len(graph_input)):
        graph.append(graph_input[i][:])
    for i in range(len(graph_input)):
        graphcopy.append(graph_input[i][:])
    pathfind = True
    vertices = list(range(len(graph)))
    path = [start]
    visited = [start]
    backtrack = [start]
    weight = 0
    backtracked = 0
    while pathfind:
        currentver = backtrack[-1]
        nextver = graph[currentver].index(min(graph[currentver]))
        if dijkstra_true and nextver == end and min(graph[currentver]) != inf:
            path.append(nextver)
            break
        if ((set(vertices) & set(visited)) == set(vertices)):
            pathfind = False
        else:
            for i in range(len(graph[currentver])):
                graph[i][currentver] = inf
            if nextver not in visited and min(graph[currentver]) != inf:
                visited.append(nextver)
                backtrack.append(nextver)
                path.append(nextver)
                weight += graphcopy[currentver][nextver]
            else:
                weight += graphcopy[currentver][backtrack[-2]]
                backtrack.pop()
                backtracked += 1
                try:
                    nextver = backtrack[-1]
                except:
                    for i in range(backtracked - 1): path.pop()
                    break
                if dijkstra_true:
                    path.pop()
                else:
                    path.append(nextver)
            if currentver == start:
                backtracked = 0
    return weight
def DWeight(graph_input, dijkstra_true=False, start=0, end=0):
    graph = []
    graphcopy = []
    for i in range(len(graph_input)):
        graph.append(graph_input[i][:])
    for i in range(len(graph_input)):
        graphcopy.append(graph_input[i][:])
    pathfind = True
    vertices = list(range(len(graph)))
    path = [start]
    visited = [start]
    backtrack = [start]
    weight = 0
    backtracked = 0
    while pathfind:
        currentver = backtrack[-1]
        nextver = graph[currentver].index(min(graph[currentver]))
        if dijkstra_true and nextver == end and min(graph[currentver]) != inf:
            path.append(nextver)
            break
        if ((set(vertices) & set(visited)) == set(vertices)):
            pathfind = False
        else:
            for i in range(len(graph[currentver])):
                graph[i][currentver] = inf
            if nextver not in visited and min(graph[currentver]) != inf:
                visited.append(nextver)
                backtrack.append(nextver)
                path.append(nextver)
            else:
                backtrack.pop()
                backtracked += 1
                try:
                    nextver = backtrack[-1]
                except:
                    for i in range(backtracked - 1): path.pop()
                    break
                if dijkstra_true:
                    path.pop()
                else:
                    path.append(nextver)
            if currentver == start:
                backtracked = 0
    for i in range(len(path)-1):
        weight += graphcopy[path[i]][path[i+1]]
    return weight

def TComplex(graph_input, dijkstra_true=False, start=0, end=0):
    time = 0
    graph = []
    for i in range(len(graph_input)):
        graph.append(graph_input[i][:])
        time += 1
    pathfind = True
    vertices = list(range(len(graph)))
    path = [start]
    visited = [start]
    backtrack = [start]
    backtracked = 0
    while pathfind:
        currentver = backtrack[-1]
        nextver = graph[currentver].index(min(graph[currentver]))
        if dijkstra_true and nextver == end and min(graph[currentver]) != inf:
            path.append(nextver)
            time += 2
            break
        if ((set(vertices) & set(visited)) == set(vertices)):
            pathfind = False
            time += 1
        else:
            for i in range(len(graph[currentver])):
                graph[i][currentver] = inf
                time += 1
            if nextver not in visited and min(graph[currentver]) != inf:
                visited.append(nextver)
                backtrack.append(nextver)
                path.append(nextver)
                time += 3
            else:
                backtrack.pop()
                backtracked += 1
                try:
                    nextver = backtrack[-1]
                    time += 1
                except:
                    for i in range(backtracked - 1):
                        path.pop()
                        time += 1
                    time += 2
                    break
                if dijkstra_true:
                    path.pop()
                    time += 1
                else:
                    path.append(nextver)
                    time += 1
                time += 6
            if currentver == start:
                backtracked = 0
                time += 1
            time += 4
        time += 5
    time += 9
    return time

def SComplex(graph_input, dijkstra_true=False, start=0, end=0):
    space = 0
    graph = []
    for i in range(len(graph_input)):
        graph.append(graph_input[i][:])
    pathfind = True
    vertices = list(range(len(graph)))
    path = [start]
    visited = [start]
    backtrack = [start]
    backtracked = 0
    while pathfind:
        currentver = backtrack[-1]
        nextver = graph[currentver].index(min(graph[currentver]))
        if dijkstra_true and nextver == end and min(graph[currentver]) != inf:
            path.append(nextver)
            break
        if ((set(vertices) & set(visited)) == set(vertices)):
            pathfind = False
        else:
            for i in range(len(graph[currentver])):
                graph[i][currentver] = inf
            if nextver not in visited and min(graph[currentver]) != inf:
                visited.append(nextver)
                backtrack.append(nextver)
                path.append(nextver)
            else:
                backtrack.pop()
                backtracked += 1
                try:
                    nextver = backtrack[-1]
                except:
                    for i in range(backtracked - 1): path.pop()
                    break
                if dijkstra_true:
                    path.pop()
                else:
                    path.append(nextver)
            if currentver == start:
                backtracked = 0
    space += sys.getsizeof(graph)
    space += sys.getsizeof(pathfind)
    space += sys.getsizeof(vertices)
    space += sys.getsizeof(path)
    space += sys.getsizeof(visited)
    space += sys.getsizeof(backtrack)
    space += sys.getsizeof(backtracked)
    return space

######################_______________KRUSKAL MST
def COMgetPairs(graph, size):
    pairs = []
    for src in range(size):
        for dest in range(src, size):
            weight = graph[src][dest]
            if weight == inf:
                continue
            else:
                pairs.append([weight, src, dest])
    return pairs

def COMbuildGraph(pairs, size):
    graph = [[inf for i in range(0, size)] for x in range(0, size)]

    for pair in pairs:
        graph[pair[1]][pair[2]] = pair[0]
        graph[pair[2]][pair[1]] = pair[0]

    return graph

def SCOMkruskal(graph):
    space = 0
    original_graph = graph
    vertices = len(graph)
    k_pairs = []
    weight_pairs = COMgetPairs(original_graph, vertices)

    weight_pairs.sort(key=lambda x: x[0])
    disjoint_set = [[x] for x in range(vertices)]
    location_set = list(range(vertices))

    while len(disjoint_set) > 1:

        try:
            pair = weight_pairs.pop(0)
        except:
            print("Graph has a node that cannot be visited")
            break

        if not (pair[1] in disjoint_set[location_set[pair[2]]]):
            k_pairs.append(pair)

            disjoint_set[location_set[pair[1]]] = disjoint_set[location_set[pair[1]]] + disjoint_set[
                location_set[pair[2]]]
            disjoint_set.pop(location_set[pair[2]])

            for location in range(location_set[pair[2]], len(disjoint_set)):
                for x in disjoint_set[location]:
                    location_set[x] = location_set[x] - 1

            for location in disjoint_set[location_set[pair[1]]]:
                location_set[location] = location_set[pair[1]]

    space += sys.getsizeof(original_graph)
    space += sys.getsizeof(vertices)
    space += sys.getsizeof(k_pairs)
    space += sys.getsizeof(weight_pairs)
    space += sys.getsizeof(disjoint_set)
    space += sys.getsizeof(location_set)
    return space

def TCOMkruskal(graph):
    time = 0
    original_graph = graph
    vertices = len(graph)
    k_pairs = []
    weight_pairs = COMgetPairs(original_graph, vertices)
    weight_pairs.sort(key=lambda x: x[0])
    disjoint_set = [[x] for x in range(vertices)]
    location_set = list(range(vertices))
    while len(disjoint_set) > 1:
        try:
            pair = weight_pairs.pop(0)
            time += 1
        except:
            time += 1
            break
        if not (pair[1] in disjoint_set[location_set[pair[2]]]):
            k_pairs.append(pair)
            disjoint_set[location_set[pair[1]]] = disjoint_set[location_set[pair[1]]] + disjoint_set[location_set[pair[2]]]
            disjoint_set.pop(location_set[pair[2]])
            for location in range(location_set[pair[2]], len(disjoint_set)):
                for x in disjoint_set[location]:
                    location_set[x] = location_set[x] - 1
                    time += 1
                time += 1
            for location in disjoint_set[location_set[pair[1]]]:
                location_set[location] = location_set[pair[1]]
                time += 1
            time += 5
        time += 3
    time += 8
    return time

######################_______________DIJKSTRA SPT
class COMpstack():
    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value)
        self.values.sort(key=lambda x: x[1],
                         reverse=True)  # Sorts in a decending order based on the index 1 of the value, which is weight

    def pop(self):
        return self.values.pop()

    def isEmpty(self):
        return (not self.values)

def SCOMdijkstra(graph, start):
    space = 0
    not_visited = COMpstack()
    not_visited.append((start, 0))
    vertices = len(graph)
    pairs = []

    result = list(map(lambda x: 0 if x == start else inf, [x for x in range(vertices)]))

    while not not_visited.isEmpty():

        current_node = not_visited.pop()

        for edge in range(vertices):

            if graph[current_node[0]][edge] == inf:
                continue

            current_weight = current_node[1] + graph[current_node[0]][edge]

            if current_weight < result[edge]:

                result[edge] = current_weight
                not_visited.append((edge, current_weight))

                for pair in range(len(pairs)):
                    if edge == pairs[pair][1] or edge == pairs[pair][2]:
                        pairs.pop(pair)
                        break
                pairs.append([graph[current_node[0]][edge], edge, current_node[0]])
            else:
                continue

    space += sys.getsizeof(not_visited)
    space += sys.getsizeof(vertices)
    space += sys.getsizeof(pairs)
    space += sys.getsizeof(result)
    return space

def TCOMdijkstra(graph, start):
    time = 0
    not_visited = COMpstack()
    not_visited.append((start, 0))
    vertices = len(graph)
    pairs = []
    result = list(map(lambda x: 0 if x == start else inf, [x for x in range(vertices)]))
    while not not_visited.isEmpty():
        current_node = not_visited.pop()
        for edge in range(vertices):
            if graph[current_node[0]][edge] == inf:
                time += 1
                continue
            current_weight = current_node[1] + graph[current_node[0]][edge]
            if current_weight < result[edge]:
                result[edge] = current_weight
                not_visited.append((edge, current_weight))
                for pair in range(len(pairs)):
                    time += 1
                    if edge == pairs[pair][1] or edge == pairs[pair][2]:
                        time += 2
                        pairs.pop(pair)
                        break
                pairs.append([graph[current_node[0]][edge], edge, current_node[0]])
                time += 4
            else:
                time += 1
                continue
            time += 4
        time += 2
    time += 5
    return time
########################################################################################################################
########################################################################################################################