import numpy as np
import matplotlib.pyplot as plt

windvals={-4:1, -3:1, -2:1, -1:2, 0:2,1:1}
alpha=0.5
epsilon=0.1
def getNextState(curr_state, action):

    horiz_disp, vert_disp=curr_state

    next_horiz_disp=horiz_disp
    next_vert_disp=vert_disp

    if horiz_disp in windvals.keys():
        next_vert_disp+=windvals[horiz_disp] 

    if action == 0:
        next_horiz_disp+=1
    elif action==1:
        next_vert_disp+=1
        next_horiz_disp+=1
    elif action==2:
        next_vert_disp+=1
    elif action==3:
        next_horiz_disp-=1
        next_vert_disp+=1
    elif action==4:
        next_horiz_disp-=1
    elif action==5:
        next_horiz_disp-=1
        next_vert_disp-=1
    elif action==6:
        next_vert_disp-=1
    else:
        next_vert_disp-=1
        next_horiz_disp+=1

    if next_horiz_disp > 2:
        next_horiz_disp=2
    if next_horiz_disp < -7:
        next_horiz_disp=-7
    if next_vert_disp >3:
        next_vert_disp=3
    if next_vert_disp < -3:
        next_vert_disp=-3 

    newstate=(next_horiz_disp,next_vert_disp)
    return newstate

def train():
    timesteps=10000
    q=[{},{},{},{},{},{},{},{}]
    x=[]
    y=[]
    t=0
    episodes=0

    for a in range(8):

        for i in range(-7,3):
            for j in range(-3,4):
                q[a][(i,j)]=0
    while t<timesteps:
        
        # new episode

        

        curr_state=(-5,0)

        while(curr_state[0] !=0 or curr_state[1] !=0):
            # print(curr_state)

            current_values=[]
            for a in range(8):
                current_values.append(q[a][curr_state])
            
            prospective_values=[]
            for a in range(8):
                ns=getNextState(curr_state,a)
                
                valprime=[]
                for aprime in range(8):
                    valprime.append(q[aprime][ns])
                
                p=np.random.random()

                if p>epsilon:
                    qprime=np.max(valprime)
                else:
                    aprime=np.random.random_integers(0,7)
                    qprime=valprime[aprime]


                value=q[a][curr_state]+alpha*(-1 + qprime-q[a][curr_state])

                prospective_values.append(value)
            
            p=np.random.random()
            if p>epsilon:
                action=np.argmax(current_values)
                #print(action)
                q[action][curr_state]=prospective_values[action]
                curr_state=getNextState(curr_state,action)
            else:
                action=np.random.random_integers(0,7)
                q[action][curr_state]=prospective_values[action]
                # print(action, curr_state)  
                curr_state=getNextState(curr_state,action)
                

            x.append(t)
            t+=1
            y.append(episodes)
        episodes+=1
        # print(q)

    return (x,y)

x,y=train()

for _ in range(10):
    x1,y1=train()
    x=[sum(a) for a in zip(x1,x)]
    y=[sum(x) for x in zip(y1,y)]

y=[x/11.0 for x in y]
x=[x/11.0 for x in x]

plt.figure()
plt.plot(x,y,label="Sarsa With King's Moves")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("No. of episodes")
plt.show()








