import argparse
parser = argparse.ArgumentParser()
import numpy as np

if __name__ == "__main__":
    visited=[]
    parser.add_argument("--grid",type=str)
    parser.add_argument("--value_policy")
    args = parser.parse_args()

    grid = np.loadtxt(args.grid, dtype=int)

    f = open(args.value_policy, "r")

    v=[]
    a=[]

    for line in f.read().split("\n"):
        if line != "":
            vals=line.split(" ")
            v.append(float(vals[0]))
            a.append(float(vals[1]))
