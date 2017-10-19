import numpy as np

grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']], dtype='S4')


start = (0,0)
goal = (3,2)

nodes = []          
distances = dict()
for x in range(0, np.shape(grid)[0]):
    for y in range(0,np.shape(grid)[1]):
        n = (x,y)
        nodes.append(n)
        distances[n] = dict()
        if(x > 0):
            neighbor = (x-1,y)
            if grid[neighbor[0], neighbor[1]] == 'o':
                distances[n][neighbor] = 1
                
        if(x < np.shape(grid)[0] - 1):
            neighbor = (x+1,y)
            if grid[neighbor[0], neighbor[1]] == 'o':
                distances[n][neighbor] = 1

        if(y > 0):
            neighbor = (x,y-1)
            if grid[neighbor[0], neighbor[1]] == 'o':
                distances[n][neighbor] = 1

        if(y < np.shape(grid)[1] - 1):
            neighbor = (x,y+1)
            if grid[neighbor[0], neighbor[1]] == 'o':
                distances[n][neighbor] = 1

unvisited = {node: None for node in nodes} #using None as +inf
visited = {}
current = goal
currentDistance = 0
unvisited[current] = currentDistance

while True:
    for neighbour, distance in distances[current].items():
        if neighbour not in unvisited: continue
        newDistance = currentDistance + distance
        if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
            unvisited[neighbour] = newDistance
    visited[current] = currentDistance
    del unvisited[current]
    if not unvisited: break
    candidates = [node for node in unvisited.items() if node[1]]
    if not candidates:
        break
    current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]

#compute shortest path
current = start
path = []
i = 0
path.append(current)
while current != goal  and i < np.shape(grid)[0] * np.shape(grid[1]):
    i = i + 1
    best_dist = 1e10
    best_neighbour = (-1,-1)
    for neighbour, d in distances[current].items():
        if visited.has_key(neighbour):
            distance = visited[neighbour]            
            if distance < best_dist:
                best_dist = distance
                best_neighbour = neighbour
    current = best_neighbour
    path.append(current)
    
print path