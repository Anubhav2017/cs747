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
    elif action==2:
        next_horiz_disp-=1
    else:
        next_vert_disp-=1

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


def qlearning():

    timesteps=10000
    q=[{},{},{},{}]
    x=[]
    y=[]
    t=0
    episodes=0

    for a in range(4):

        for i in range(-7,3):
            for j in range(-3,4):
                q[a][(i,j)]=0
    while t<timesteps:
        # new episode
        curr_state=(-5,1)

        while(curr_state[0] !=0 or curr_state[1] !=0):
   
            current_values=[]
            for a in range(4):
                current_values.append(q[a][curr_state])

            prospective_values=[]
            for a in range(4):
                ns=getNextState(curr_state,a)
                
                valprime=[]
                for aprime in range(4):
                    valprime.append(q[aprime][ns])
                
                p=np.random.random()

                qprime=np.max(valprime)
                
                value=q[a][curr_state]+alpha*(-1 + qprime-q[a][curr_state])

                prospective_values.append(value)
            # print(values)
            p=np.random.random()
            if p>epsilon:
                action=np.argmax(current_values)
                q[action][curr_state]=prospective_values[action]
                curr_state=getNextState(curr_state,action)
            else:
                action=np.random.random_integers(0,3)
                val=prospective_values[action]
                q[action][curr_state]=val
                curr_state=getNextState(curr_state,action)
                

            x.append(t)
            t+=1
            y.append(episodes)
        episodes+=1

    return (x,y)


def sarsa():
    timesteps=10000
    q=[{},{},{},{}]
    x=[]
    y=[]
    t=0
    episodes=0

    for a in range(4):

        for i in range(-7,3):
            for j in range(-3,4):
                q[a][(i,j)]=0
    while t<timesteps:
        # new episode
        curr_state=(-5,0)

        while(curr_state[0] !=0 or curr_state[1] !=0):
            # print(curr_state)

            current_values=[]
            for a in range(4):
                current_values.append(q[a][curr_state])
            
            prospective_values=[]
            for a in range(4):
                ns=getNextState(curr_state,a)
                
                valprime=[]
                for aprime in range(4):
                    valprime.append(q[aprime][ns])
                
                p=np.random.random()

                if p> epsilon:
                    qprime=np.max(valprime)
                else:
                    aprime=np.random.random_integers(0,3)
                    qprime=valprime[aprime]
                
                # print("ns",ns,"valprime",valprime,"qacs",q[a][curr_state])


                value=q[a][curr_state]+alpha*(-1 + qprime-q[a][curr_state])

                prospective_values.append(value)
            p=np.random.random()
            if p>epsilon:
                action=np.argmax(current_values)
                #print(action)
                q[action][curr_state]=prospective_values[action]
                curr_state=getNextState(curr_state,action)
                # print(action, curr_state)
                # print(q[action][curr_state]) 
            else:
                action=np.random.random_integers(0,3)
                val=prospective_values[action]
                q[action][curr_state]=val
                # print(action, curr_state)  
                curr_state=getNextState(curr_state,action)
                

            x.append(t)
            t+=1
            y.append(episodes)
        episodes+=1
        # print(q)

    return (x,y)


def expectedsarsa():
    timesteps=10000
    q=[{},{},{},{}]
    x=[]
    y=[]
    t=0
    episodes=0

    for a in range(4):

        for i in range(-7,3):
            for j in range(-3,4):
                q[a][(i,j)]=0
    while t<timesteps:
        # new episode
        curr_state=(-5,0)

        while(curr_state[0] !=0 or curr_state[1] !=0):
            
            current_values=[]
            for a in range(4):
                current_values.append(q[a][curr_state])

            prospective_values=[]
            for a in range(4):
                ns=getNextState(curr_state,a)
                
                valprime=[]
                for aprime in range(4):
                    valprime.append(q[aprime][ns])
                
                
                qprime=np.sum(valprime)*epsilon/4
                qprime+=(1-epsilon)*np.max(valprime)

                value=q[a][curr_state]+alpha*(-1 + qprime-q[a][curr_state])

                prospective_values.append(value)
            
            p=np.random.random()
            if p>epsilon:
                action=np.argmax(current_values)
                #print(action)
                q[action][curr_state]=prospective_values[action]
                curr_state=getNextState(curr_state,action)
                # print(action, curr_state) 
            else:
                action=np.random.random_integers(0,3)
                val=prospective_values[action]
                q[action][curr_state]=val
                # print(action, curr_state)  
                curr_state=getNextState(curr_state,action)
                

            x.append(t)
            t+=1
            y.append(episodes)
        episodes+=1
        # print(q)

    return (x,y)


plt.figure()

#-------------------------------------------------------
xsarsa,ysarsa=sarsa()

for _ in range(10):
    x1,y1=sarsa()
    xsarsa=[sum(a) for a in zip(x1,xsarsa)]
    ysarsa=[sum(x) for x in zip(y1,ysarsa)]

ysarsa=[x/11.0 for x in ysarsa]
xsarsa=[x/11.0 for x in xsarsa]
plt.plot(xsarsa,ysarsa, label="Sarsa")

#-----------------------------------------------------------
xq,yq=qlearning()

for _ in range(10):
    x1,y1=qlearning()
    xq=[sum(a) for a in zip(x1,xq)]
    yq=[sum(x) for x in zip(y1,yq)]

yq=[x/11.0 for x in yq]
xq=[x/11.0 for x in xq]
plt.plot(xq,yq, label="Q-learning")

#----------------------------------------------------

xesarsa,yesarsa=expectedsarsa()

for _ in range(10):
    x1,y1=expectedsarsa()
    xesarsa=[sum(a) for a in zip(x1,xesarsa)]
    yesarsa=[sum(x) for x in zip(y1,yesarsa)]

yesarsa=[x/11.0 for x in yesarsa]
xesarsa=[x/11.0 for x in xesarsa]

plt.plot(xesarsa,yesarsa,label="Expected Sarsa")

plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("No. of episodes")
plt.show()





