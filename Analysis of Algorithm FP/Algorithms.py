from random import randint
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sys import exit
from Analytics import *
inf = float('inf')

# A pair is a list containing 3 values, [weight, source node, destination node]
# A value in a pstack is a list containing 2 values [node/vertex, weight]

# Preset graphs

graph1 = [
[inf, inf, 6, 2, 3],
[inf, inf, 1, 6, 2],
[6, 1, inf, 1, 9],
[2, 6, 1, inf, 1],
[3, 2, 9, 1, inf]
]

graph2 = [
[inf, 6, inf, 8, 7], 
[6, inf, 4, 9, 9], 
[inf, 4, inf, 7, 1], 
[8, 9, 7, inf, inf], 
[7, 9, 1, inf, inf]
]

graph3 = [
    [inf, 6, 8, inf, inf],
    [6, inf, 9, 8, inf],
    [8, 9, inf, 8, inf],
    [inf, 8, 8, inf, inf],
    [inf, inf, inf, inf, inf]
]
graph4 = [
    [inf, 5, inf, 3, 5, 8],
    [5, inf, 3, 4, inf, inf],
    [inf, 3, inf, inf, 6, 2],
    [3, 4, inf, inf, 2, inf],
    [5, inf, 6, 2, inf, inf],
    [8, inf, 2, inf, inf, inf]
]

graph1_dijkstra = [
[inf,inf,inf,2,3],
[inf, inf, 1, inf, inf],
[inf, 1, inf, 3, inf],
[2, inf, 1, inf, inf],
[3, inf, inf, inf, inf]
]

graph2_dijkstra = [
[inf, 6, inf, 8, 7],
[6, inf, inf, inf, inf],
[inf,inf,inf,inf,1],
[8,inf,inf,inf,inf],
[7,inf,1,inf,inf]
]



preset_graphs = [graph1, graph2, graph3, graph4]

class pstack(): # The priority stack, basically a stack that will always, in this case, pop the smallest value (this stack is for the dijkstra function below)
    def __init__(self):
        self.values = [] # The list that stores the values in the stack, this stack accepts tuples, containing a node in index 0 and the weight or value of that node in index 1
        
    def append(self, value): # Appends a value to the stack
        self.values.append(value) # Appends to the self.values list
        self.values.sort(key=lambda x: x[1], reverse = True) # Sorts in a decending order based on the index 1 of the value, which is weight
    
    def pop(self): # Pops a value from the stack
        return self.values.pop() # Pops from self.values at the last index, which is the top of the stack
        
    def isEmpty(self): # Returns a boolean to check if the stack is empty
        return (not self.values) # checks if self.values is empty

def generateGraph(size): # Returns a random graph based on the given size parameter
    graph = [[None for i in range(0,size)] for x in range(0,size)] # O(n) # Initializes the graph None value type, to show that there are no values assigned to the paths yet
    for vertex in range(0,size): # Populates the vertices
        for edge in range(0,size): # Populates the edges, which is list of weights originating from that vertex
            rand = randint(0, 31) # Generates a random number from 0 until 30
            if rand == 0: rand = inf # Sets zeros as infinity, since zero should represent there is no weight or no path to the corresponding vertex
            if vertex == edge: # The weight of the path to the vertex itself should be set to infinity
                graph[vertex][edge] = inf 

            elif graph[edge][vertex] != None: # If there is already a weight assigned
                graph[vertex][edge] = graph[edge][vertex] # Set the weight of the opposite node the same as the current one, since this generates an undirected graph
            else:
                if rand%randint(1, 9) == 0: # Raises the chance of infinity to be set as a weight value
                    graph[vertex][edge] = inf
                else:
                    graph[vertex][edge] = rand # Sets the random number as the weight of the path from the current vertex
    return graph # returns the random graph

def printall(object): # Prints the graph, node by node
    for x in range(len(object)): # Iterates through the graph
        tabs = (len(object)%10)-len(str(x)) # Sets the amout of tabs
        print("Node {0}:{1}".format(x, " "*tabs),object[x]) # Prints the current node and its list of weighted paths

def search(array, value): # Searches a value in an array and returns its index (Linear search)
    for index in range(len(array)):
        if array[index] == value:
            return index
    return inf
    
def getPairs(graph, size): # Generates a list of pairs (pairs are explained at the top)
    pairs = [] # initializes the pairs list
    for src in range(size): # iterates through all the nodes in the given graph
        for dest in range(src, size): # iterates through all the weights in the current node
            weight = graph[src][dest] 
            if weight == inf: # Continues if the weight is infinity, since it would mean the current pair is not connected to one another
                continue
            else:
                pairs.append([weight, src, dest]) # appends the pair into the pairs list
    return pairs # returns the pairs list
    
def buildGraph(pairs, size): # The counter part to getPairs(), this function generates an adjacency matrix from a pairs list
    graph = [[inf for i in range(0,size)] for x in range(0,size)] # initializes the matrix with infinity
    
    for pair in pairs: # iterates through all the pairs in the pairs list
        graph[pair[1]][pair[2]] = pair[0]
        graph[pair[2]][pair[1]] = pair[0]

    return graph # returns the adjacency matrix

def kruskal(graph):
    original_graph = graph
    vertices = len(graph)
    k_pairs = []
    weight_pairs = getPairs(original_graph, vertices)

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

    return buildGraph(k_pairs, vertices)
        
def dijkstra(graph, start):
    not_visited = pstack()
    not_visited.append((start,0))
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
    return buildGraph(pairs, vertices)
    
def pathFind(graph_input, dijkstra_true=False, start=0, end=0):
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

        if ((set(vertices)&set(visited))==set(vertices)):
            pathfind=False
        else:
            for i in range (len(graph[currentver])):
                graph[i][currentver] = inf

            if nextver not in visited and min(graph[currentver]) != inf:
                visited.append(nextver)
                backtrack.append(nextver)
                path.append(nextver)

            else:
                backtrack.pop()
                backtracked+=1

                try:
                    nextver = backtrack[-1]
                except:
                    print("Graph has a node that cannot be visited")
                    for i in range(backtracked-1):path.pop()
                    break

                if dijkstra_true:
                    path.pop()
                else:
                    path.append(nextver)

            if currentver == start:
                backtracked = 0
    return path

def dijkstraPathFind(graph_input, start, end=0):
    graph = dijkstra(graph_input, start)
    return pathFind(graph, True, start, end)
def WdijkstraPathFind(graph_input, start, end=0):
    graph = dijkstra(graph_input, start)
    return DWeight(graph, True, start, end)
def TdijkstraPathFind(graph_input, start, end=0):
    graph = dijkstra(graph_input, start)
    return TComplex(graph, True, start, end)
def SdijkstraPathFind(graph_input, start, end=0):
    graph = dijkstra(graph_input, start)
    return SComplex(graph, True, start, end)

def graphMatrix(graph,path,weight1,weight2,weight,time,space):
    graph_edge = [] 
    graph_label={} 
    for i in range(len(graph)):
        for x in range(len(graph[i])):
            if graph[i][x] != inf:
                graph_edge.append(tuple([i,x]))
                graph_label.update({tuple([i,x]):graph[i][x]})
    
    G = nx.Graph()
    G.add_edges_from(graph_edge)
    pos=nx.spring_layout(G)
    nx.draw(G, pos, font_size= 20, node_color='lime', node_size=800, with_labels=True, label='test')
    nx.draw_networkx_edge_labels(G, pos, font_size= 20, font_color='red', edge_labels=graph_label,rotate=0)

    show_path = mpatches.Patch(label=('Path: '+str(path)))
    show_weight1 = mpatches.Patch(label=('Pathfind(DFS) weight: ' + str(weight1)))
    show_weight2 = mpatches.Patch(label=('Return(Dijkstra) weight: ' + str(weight2)))
    show_weight = mpatches.Patch(label=('Total weight: ' + str(weight)))
    show_time = mpatches.Patch(label=('Time Complexity: ' + str(time/10) + ' ms'))
    show_space = mpatches.Patch(label=('Space Complexity: ' + str(space) + ' bytes'))
    plt.legend(handles=[show_path, show_weight1, show_weight2, show_weight, show_time, show_space],loc=2,prop={'size':16})
    plt.show()

def integerChecker(text):
    while True:
        try:
            integer = int(input(text))
            break
        except:
            print("Please input an integer")
            continue
    return integer

def menuInputChecker(max, text):
    while True:
        try:
            user_input = int(input(text+"[0] Exit\n"))
        except:
            print("Please input an integer")
            continue
        if user_input < (-1) or user_input > max:
            print("Please select an available option")
            continue
        break
    if user_input == 0: exit()
    return user_input

def selectGraph():
    user_input = menuInputChecker(2, "Menu Options\n[1] Generate Graph\n[2] Preset\n")
        
    graph = []
    if user_input == 1:
        size = integerChecker("Enter number of nodes:")
        graph = generateGraph(size)
    elif user_input == 2:
        print("Avalilable Graphs")
        graph_string = ""
        for graph_index in range(len(preset_graphs)):
            current_graph = "Graph {}".format(graph_index + 1)
            print(current_graph)
            printall(preset_graphs[graph_index])
            graph_string += "[{0}] {1}\n".format(graph_index + 1, current_graph)
            
        graph_selection = menuInputChecker(len(preset_graphs), "Please select a Graph\n"+graph_string)
        graph = preset_graphs[graph_selection-1]
    return graph

def main():
    graphcopy = selectGraph()
    for i in range(len(graphcopy)):
        print(graphcopy[i])
    path = []
    while True:
        graph = graphcopy
        starting_point = 0
        user_input = menuInputChecker(4, "[1] Plot unmodified graph\n[2] Plot MST (Kruskal) graph\n[3] Plot SPT (Dijkstra) graph\n[4] Select different graph\n")
        if user_input == 1: # Plot unmodified graph
            path = pathFind(graph, False)
            path = path + dijkstraPathFind(graph, path[-1], 0)[1:]
        elif user_input == 2: # Plot MST (Kruskal) graph
            graph = (kruskal(graph))
            path = pathFind((graph), False)
            path = path + dijkstraPathFind(graph, path[-1], 0)[1:]
        elif user_input == 3: # Plot SPT (Dijkstra) graph
            starting_point = integerChecker("Please input a starting point ")
            graph = (dijkstra(graph, starting_point))
            path = pathFind(graph,False)
            path = path + dijkstraPathFind(graph, path[-1], 0)[1:]
        elif user_input == 4: # Select different graph
            graphcopy = selectGraph()

        # Analysis
        if user_input != 4:
            weight1 = Weight(graph, False)
            weight2 = WdijkstraPathFind(graph,path[-1],0)
            weight = weight1 + weight2
            time = 0
            space = 0
            if user_input == 1:
                time = TComplex(graph, False) + TdijkstraPathFind(graph, path[-1], 0)
                space = SComplex(graph, False) + SdijkstraPathFind(graph, path[-1], 0)
            elif user_input == 2:
                time =  TCOMkruskal(graph) + TComplex(graph, False) + TdijkstraPathFind(graph, path[-1], 0)
                space = SCOMkruskal(graph) + SComplex(graph, False) + SdijkstraPathFind(graph, path[-1], 0)
            elif user_input == 3:
                time = TCOMdijkstra(graph,starting_point) + TComplex(graph, False) + TdijkstraPathFind(graph, path[-1], 0)
                space = SCOMdijkstra(graph,starting_point) + SComplex(graph, False) + SdijkstraPathFind(graph, path[-1], 0)
            graphMatrix(graph, path, weight1, weight2, weight, time, space)
            print('Path: ' + str(path) + '\nPathfind(DFS) weight: ' + str(weight1) +
                  '\nReturn(Dijkstra) weight: ' + str(weight2) + '\nTotal weight: ' + str(weight) +
                  '\nTime Complexity: ' + str(time / 10) + ' ms' + '\nSpace Complexity: ' + str(space) + ' bytes')
main()
