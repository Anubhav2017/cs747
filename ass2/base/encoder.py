import sys
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
import numpy as np

visited=[]
curr_state=0

trans_statements=[]

def explore(grid, i, j):
    global visited
    visited.append((i,j))
    global curr_state
    mystate=curr_state

    if grid[i][j] == 3:
        print("end",curr_state)
        return

    if ((grid[i+1][j] != 1)  and (i<nr-1) and ((i+1,j) not in visited)):
        trans_statements.append(("transition {} {} {} {} {}").format(mystate, 0, curr_state+1, 1,1))
        curr_state+=1
        explore(grid,i+1,j)
    if ((grid[i][j+1] != 1) and (grid[i][j] !=3) and (j<nc-1) and ((i,j+1) not in visited)):
        trans_statements.append(("transition {} {} {} {} {}").format(mystate, 1, curr_state+1, 1,1))
        curr_state+=1
        explore(grid, i,j+1)
    if ((grid[i-1][j] != 1) and (grid[i][j] !=3) and  (i>1) and ((i-1,j) not in visited)):
        trans_statements.append(("transition {} {} {} {} {}").format(mystate, 2, curr_state+1, 1,1))
        curr_state+=1
        explore(grid, i-1,j)
    if ((grid[i][j-1] != 1) and (grid[i][j] !=3) and (j>1) and ((i,j-1) not in visited)):
        trans_statements.append(("transition {} {} {} {} {}").format(mystate, 3, curr_state+1, 1,1))
        curr_state+=1
        explore(grid, i,j-1)
    return

if __name__ == "__main__":
    visited=[]
    parser.add_argument("--grid",type=str)
    args = parser.parse_args()

    grid = np.loadtxt(args.grid, dtype=int)
    count=0

    nr = len(grid)
    nc = len(grid[0])
    x,y = 0,0

    for i in range(nr):
        for j in range(nc):
            if grid[i][j] != 1:
                count += 1

    print("numStates",count)
    print("numActions",4)
    print("start",0)


    for i in range(nr):
        for j in range(nc):
            if grid[i][j] == 2:
                x,y = i,j
                visited.append((i,j))
                curr_state=0
                
                if ((grid[i+1][j] != 1) and (i<nr-1) and ((i+1,j) not in visited) ):
                    
                    trans_statements.append(("transition {} {} {} {} {}").format(curr_state, 0, curr_state+1, 1.0,1))
                    curr_state+=1
                    explore(grid,i+1,j)
                if ((grid[i][j+1] != 1) and (j<nc-1) and ((i,j+1) not in visited)):
                    
                    trans_statements.append(("transition {} {} {} {} {}").format(curr_state, 1, curr_state+1, 1.0,1))
                    curr_state+=1
                    explore(grid, i,j+1)
                if ((grid[i-1][j] != 1) and (i>1) and ((i-1,j) not in visited)):
                    
                    trans_statements.append(("transition {} {} {} {} {}").format(curr_state, 2, curr_state+1, 1.0,1))
                    curr_state+=1
                    explore(grid, i-1,j)
                if ((grid[i][j-1] != 1) and (j>1) and ((i,j-1) not in visited)):
                    
                    trans_statements.append(("transition {} {} {} {} {}").format(curr_state, 3, curr_state+1, 1.0,1))
                    curr_state+=1
                    explore(grid, i,j-1)

    for i in range(len(trans_statements)):
        print(trans_statements[i])

    print("mdptype episodic")
    print("discount  0.9")

       

       