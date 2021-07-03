import sys
import argparse
parser = argparse.ArgumentParser()
import numpy as np

states={}

if __name__ == "__main__":
    visited=[]
    parser.add_argument("--grid",type=str)
    args = parser.parse_args()

    grid = np.loadtxt(args.grid, dtype=int)

    nr = len(grid)
    nc = len(grid[0])

    count=0

    for i in range(nr):
        for j in range(nc):

            if grid[i][j] != 1:
                states[(i,j)]=count
                count+=1

            if grid[i][j] == 2:
                print("start",states[(i,j)]) 

            if grid[i][j] == 3:
                print("end", states[(i,j)])

    print("numStates",count)
    print("numActions",4)

    for i in range(nr):
        for j in range(nc):
            if(grid[i][j]==0 or grid[i][j]==2):
                s0=states[(i,j)]
                if(grid[i+1][j]==0 or grid[i+1][j]==2):
                    s1=states[(i+1,j)]
                    print(("transition {} {} {} {} {}").format(s0, 0,s1, -1.0,1.0))

                if(grid[i][j+1]==0 or grid[i][j+1]==2):
                    s1=states[(i,j+1)]
                    print(("transition {} {} {} {} {}").format(s0, 1,s1, -1.0,1.0))

                if(grid[i-1][j]==0 or grid[i-1][j]==2):
                    s1=states[(i-1,j)]
                    print(("transition {} {} {} {} {}").format(s0, 2,s1, -1.0,1.0))

                if(grid[i][j-1]==0 or grid[i][j-1]==2):
                    s1=states[(i,j-1)]
                    print(("transition {} {} {} {} {}").format(s0, 3,s1, -1.0,1.0))

                if(grid[i+1][j]==3):
                    s1=states[(i+1,j)]
                    print(("transition {} {} {} {} {}").format(s0, 0,s1, 100.0,1.0))

                if(grid[i][j+1]==3):
                    s1=states[(i,j+1)]
                    print(("transition {} {} {} {} {}").format(s0, 1,s1, 100.0,1.0))

                if(grid[i-1][j]==3):
                    s1=states[(i-1,j)]
                    print(("transition {} {} {} {} {}").format(s0, 2,s1, 100.0,1.0))

                if(grid[i][j-1]==3):
                    s1=states[(i,j-1)]
                    print(("transition {} {} {} {} {}").format(s0, 3,s1, 100.0,1.0))
                
                
                                               


    print("mdptype episodic")
    print("discount  1.0")

       

       