import argparse
parser = argparse.ArgumentParser()
import numpy as np
states={}

mdp=[]

if __name__ == "__main__":
    visited=[]
    parser.add_argument("--grid",type=str)
    parser.add_argument("--value_policy")
    args = parser.parse_args()

    grid = np.loadtxt(args.grid, dtype=int)

    nr = len(grid)
    nc = len(grid[0])

    f = open(args.value_policy, "r")

    v=[]
    a=[]

    for line in f.read().split("\n"):
        if line != "":
            vals=line.split(" ")
            v.append(float(vals[0]))
            a.append(float(vals[1]))

    start=0
    end=0
    count=0

    for i in range(nr):
        for j in range(nc):

            if grid[i][j] != 1:
                states[(i,j)]=count
                count+=1

            if grid[i][j] == 2:
                start=states[(i,j)] 

            if grid[i][j] == 3:
                end=states[(i,j)]

    mdp=[{} for i in range(count)]

    for i in range(nr):
        for j in range(nc):
            if(grid[i][j]!=1):
                s0=states[(i,j)]
                if grid[i+1][j] != 1:
                    s1=states[(i+1,j)]
                    mdp[s0][0]=s1

                if grid[i][j+1] != 1:
                    s1=states[(i,j+1)]
                    mdp[s0][1]=s1

                if grid[i-1][j] != 1:
                    s1=states[(i-1,j)]
                    mdp[s0][2]=s1

                if grid[i][j-1] != 1:
                    s1=states[(i,j-1)]
                    mdp[s0][3]=s1

    curr_state=start
    it=0 
    while(curr_state != end or it>20):
        action=a[curr_state]
        if action==0:
            print('S',end=' ')
        elif action == 1:
            print('E',end=' ')
        elif action == 2:
            print('N',end=' ')
        elif action == 3:
            print('W',end=' ')
        curr_state=mdp[curr_state][action]



    
