#! /usr/bin/python
import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import pulp as pulp

def hpi(mdpfile):
    infile = open(mdpfile, 'r') 
    # lines = infile.readlines('\n')
    numstates=0
    numactions=0
    start=0
    end=-1
    discount=0.0
    mdptype="continuing"
    s0=[]
    s1=[]
    a=[]
    r=[]
    t=[]


    for line in infile:

        line =line.rstrip('\n')
        items=line.split(" ")
        if(items[0] == "numStates"):
            numstates=int(items[1])
        elif(items[0]=="numActions"):
            numactions=int(items[1])
        elif(items[0]=="start"):
            start=int(items[1])
        elif(items[0]=="end"):
            end=int(items[1])
        elif(items[0]=="discount"):
            discount=float(items[2])
        elif(items[0]=="mdptype"):
            mdptype=items[1]
        else:
            s0.append(int(items[1]))
            a.append(int(items[2]))
            s1.append(int(items[3]))
            r.append(float(items[4]))
            t.append(float(items[5]))

    data_ns=[{} for i in range(numstates)]
    data_T=[{} for i in range(numstates)]
    data_R=[{} for i in range(numstates)]

    

    for i in range(numstates):
        for j in range(numactions):
            data_ns[i][j]=[]
            data_T[i][j]=[]
            data_R[i][j]=[]

    for i in range(len(s0)):
        curr_state=s0[i]
        ns=s1[i]
        action=a[i]
        reward=r[i]
        tp=t[i]
        # print(action)
        data_ns[curr_state][action].append(ns)
        data_R[curr_state][action].append(reward)
        data_T[curr_state][action].append(tp)
      
    # print(data_T)
    # print(data_R)
    v=[random.random() for i in range(numstates)]

    current_policy=[0 for _ in range(numstates)]

    for i in range(numstates):
        actions=list(data_ns[i].keys())
        current_policy[i]=actions[0]

    
    newpolicy=current_policy[:]

    A=np.zeros((numstates,numstates))
    B=np.zeros((numstates))

    for i in range(numstates):
        next_states=data_ns[i][current_policy[i]]
        bval=0
        A[i][i]=1
        for j in range(len(next_states)):
            bval+=data_R[i][current_policy[i]][j]*data_T[i][current_policy[i]][j]
            ns=data_ns[i][current_policy[i]][j]
            A[i][ns] -= discount*data_T[i][current_policy[i]][j]

        B[i]=bval

    v=np.linalg.inv(A).dot(B)

    
    for i in range(numstates):
        actions=list(data_ns[i].keys())
        a=current_policy[i]

        next_states=data_ns[i][a]
        value=0

        for k in range(len(next_states)):

            T= data_T[i][a][k]
            R= data_R[i][a][k]
            vt1=v[next_states[k]]
            value+=T*(R+discount*vt1)
        currvalue=value

        allvalues=[]

        for j in actions:

            next_states=data_ns[i][j]
            value=0

            for k in range(len(next_states)):

                T= data_T[i][j][k]
                R= data_R[i][j][k]
                vt1=v[next_states[k]]
                value+=T*(R+discount*vt1)
            allvalues.append(value)
        if(currvalue<np.max(allvalues)):
            newpolicy[i]=list(data_ns[i].keys())[np.argmax(allvalues)]


    while(newpolicy != current_policy):
        current_policy=newpolicy[:]

        A=np.zeros((numstates,numstates))
        B=np.zeros((numstates))

        for i in range(numstates):
            next_states=data_ns[i][current_policy[i]]
            bval=0
            A[i][i]=1
            for j in range(len(next_states)):
                bval+=data_R[i][current_policy[i]][j]*data_T[i][current_policy[i]][j]
                ns=data_ns[i][current_policy[i]][j]
                A[i][ns] -= discount*data_T[i][current_policy[i]][j]

            bval*=discount
            B[i]=bval

        v=np.linalg.inv(A).dot(B)
        
        for i in range(numstates):
            actions=list(data_ns[i].keys())
            a=current_policy[i]

            next_states=data_ns[i][a]
            value=0

            for k in range(len(next_states)):

                T= data_T[i][a][k]
                R= data_R[i][a][k]
                vt1=v[next_states[k]]
                value+=T*(R+discount*vt1)
            currvalue=value

            allvalues=[]

            for j in actions:

                next_states=data_ns[i][j]
                value=0

                for k in range(len(next_states)):

                    T= data_T[i][j][k]
                    R= data_R[i][j][k]
                    vt1=v[next_states[k]]
                    value+=T*(R+discount*vt1)
                allvalues.append(value)
            if(currvalue<np.max(allvalues)):
                newpolicy[i]=list(data_ns[i].keys())[np.argmax(allvalues)]
            # print(allvalues)    
    # print(newpolicy)

    A=np.zeros((numstates,numstates))
    B=np.zeros((numstates))

    for i in range(numstates):
        next_states=data_ns[i][newpolicy[i]]
        bval=0
        A[i][i]=1
        for j in range(len(next_states)):
            bval+=data_R[i][newpolicy[i]][j]*data_T[i][newpolicy[i]][j]
            ns=data_ns[i][newpolicy[i]][j]
            A[i][ns] -= discount*data_T[i][newpolicy[i]][j]

        B[i]=bval
    # print(A)
    # print(B)
    v=np.linalg.inv(A).dot(B)

    for i in range(numstates):
        print(v[i], newpolicy[i])

def vi(mdpfile):
    infile = open(mdpfile, 'r') 
    numstates=0
    numactions=0
    start=0
    end=-1
    discount=0.0
    mdptype="continuing"
    s0=[]
    s1=[]
    a=[]
    r=[]
    t=[]

    for line in infile:

        line =line.rstrip('\n')
        items=line.split(" ")
        if(items[0] == "numStates"):
            numstates=int(items[1])
        elif(items[0]=="numActions"):
            numactions=int(items[1])
        elif(items[0]=="start"):
            start=int(items[1])
        elif(items[0]=="end"):
            end=int(items[1])
        elif(items[0]=="discount"):
            discount=float(items[2])
        elif(items[0]=="mdptype"):
            mdptype=items[1]
        else:
            s0.append(int(items[1]))
            a.append(int(items[2]))
            s1.append(int(items[3]))
            r.append(float(items[4]))
            t.append(float(items[5]))

    data_ns=[{} for i in range(numstates)]
    data_T=[{} for i in range(numstates)]
    data_R=[{} for i in range(numstates)]

    for i in range(numstates):
        for j in range(numactions):
            data_ns[i][j]=[]
            data_T[i][j]=[]
            data_R[i][j]=[]

    for i in range(len(s0)):
        curr_state=s0[i]
        ns=s1[i]
        action=a[i]
        reward=r[i]
        tp=t[i]
        data_ns[curr_state][action].append(ns)
        data_R[curr_state][action].append(reward)
        data_T[curr_state][action].append(tp)

    v=[random.random() for i in range(numstates)]

    next_v=[0 for i in range(numstates)]
    for i in range(numstates):
        next_values=[]
        actions=data_ns[i].keys()
        for j in actions:
            next_states=data_ns[i][j]

            value=0
            for k in range(len(next_states)):
                T= data_T[i][j][k]
                R= data_R[i][j][k]
                vt1=v[next_states[k]]
                value+=T*(R+discount*vt1)

            next_values.append(value)

        next_value=np.max(next_values)
        next_v[i]=next_value


    while(abs(np.sum(np.array(v) - np.array(next_v))) > 1e-10):
        v=next_v

        next_v=[0 for i in range(numstates)]
        for i in range(numstates):
            next_values=[]
            actions=data_ns[i].keys()
            for j in actions:
                next_states=data_ns[i][j]
                value=0
                for k in range(len(next_states)):
                    T= data_T[i][j][k]
                    R= data_R[i][j][k]
                    vt1=v[next_states[k]]
                    value+=T*(R+discount*vt1)

                # print(value)
                next_values.append(value)

            next_value=np.max(next_values)
            next_v[i]=next_value

    v=next_v[:]
    policy=[0 for _ in range(numstates)]

    for i in range(numstates):
        actions=list(data_ns[i].keys())
        allvalues=[]

        for j in actions:

            next_states=data_ns[i][j]
            value=0

            for k in range(len(next_states)):

                T= data_T[i][j][k]
                R= data_R[i][j][k]
                vt1=v[next_states[k]]
                value+=T*(R+discount*vt1)
            allvalues.append(value)

        policy[i]=list(data_ns[i].keys())[np.argmax(allvalues)]
    
    for i in range(numstates):
        print(v[i], policy[i])



def lp(mdpfile):
    infile = open(mdpfile, 'r') 
    # lines = infile.readlines('\n')
    numstates=0
    numactions=0
    start=0
    end=-1
    discount=0.0
    mdptype="continuing"
    s0=[]
    s1=[]
    a=[]
    r=[]
    t=[]


    for line in infile:

        line =line.rstrip('\n')
        items=line.split(" ")
        if(items[0] == "numStates"):
            numstates=int(items[1])
        elif(items[0]=="numActions"):
            numactions=int(items[1])
        elif(items[0]=="start"):
            start=int(items[1])
        elif(items[0]=="end"):
            end=int(items[1])
        elif(items[0]=="discount"):
            discount=float(items[2])
        elif(items[0]=="mdptype"):
            mdptype=items[1]
        else:
            s0.append(int(items[1]))
            a.append(int(items[2]))
            s1.append(int(items[3]))
            r.append(float(items[4]))
            t.append(float(items[5]))

    data_ns=[{} for i in range(numstates)]
    data_T=[{} for i in range(numstates)]
    data_R=[{} for i in range(numstates)]

    for i in range(numstates):
        for j in range(numactions):
            data_ns[i][j]=[]
            data_T[i][j]=[]
            data_R[i][j]=[]

    for i in range(len(s0)):
        curr_state=s0[i]
        ns=s1[i]
        action=a[i]
        reward=r[i]
        tp=t[i]
        # print(action)
        data_ns[curr_state][action].append(ns)
        data_R[curr_state][action].append(reward)
        data_T[curr_state][action].append(tp)

    prob= pulp.LpProblem('problem', pulp.LpMaximize)

    lpvars=[]

    for i in range(numstates):

        lpvars.append(pulp.LpVariable("vs{}".format(i)))

    obj_fn=0
    for i in range(numstates):
        obj_fn -= lpvars[i]
    
    prob+= obj_fn

    for i in range(numstates):
        next_values=[]
        actions=data_ns[i].keys()
        for j in actions:
            next_states=data_ns[i][j]
            value=0
            for k in range(len(next_states)):
            # print("i={}, j={}, k={}".format(i,j,k))    
                T= data_T[i][j][k]
                R= data_R[i][j][k]
                vt1=lpvars[next_states[k]]
                value+=T*(R+discount*vt1)

            # print(value)
            prob+= lpvars[i] >= value

    prob.solve()
    v=[]

    for i in range(numstates):
        v.append(lpvars[i].varValue)
    
    # print(v)
    policy=[0 for _ in range(numstates)]

    for i in range(numstates):
        actions=list(data_ns[i].keys())
        allvalues=[]

        for j in actions:

            next_states=data_ns[i][j]
            value=0

            for k in range(len(next_states)):

                T= data_T[i][j][k]
                R= data_R[i][j][k]
                vt1=v[next_states[k]]
                value+=T*(R+discount*vt1)
            allvalues.append(value)

        policy[i]=list(data_ns[i].keys())[np.argmax(allvalues)]
    
    for i in range(numstates):
        print(v[i], policy[i])




if __name__ == "__main__":

    parser.add_argument("--mdp",type=str,default=0.9)
    parser.add_argument("--algorithm",type=str,default="vi")
        
    args = parser.parse_args()

    if (args.algorithm == "vi"):
        vi(args.mdp)

    elif(args.algorithm == "hpi"):
        hpi(args.mdp)
    else:
        lp(args.mdp)


