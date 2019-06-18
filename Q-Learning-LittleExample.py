import numpy as np
import pandas as pd
import time

np.random.seed(2) ##reproducible

N_STATES=6 ##the length of the 1 dimensional world
ACTIONS=['left','right'] ##avialable actions
EPSILON=0.9 ##greedy police
ALPHA=0.1  ##learning rate
LAMBDA=0.9 ##discount factor
MAX_EPISODES=13 ##maximum epsiodes
FRESH_TIME=0.1 ##fresh time for one move

def bulid_q_table(n_states,actions):
    table=pd.DataFrame(np.zeros((n_states,len(actions))),columns=actions,)  ##q_table initial values
    # print(table)
    return table

def choose_action(state,q_table):
    '''This is how to choose an action'''
    state_actions=q_table.iloc[state,:]
    if (np.random.uniform()>EPSILON) or (state_actions.all()==0): ##act non-greedy or state-actions equal 0
        action_name=np.random.choice(ACTIONS)  ##return right or left
    else:
        action_name=state_actions.idxmax()  ##以前是argmax()  现在改成了 idxmax
    return action_name

def get_env_feedback(S,A):
    '''This is how agent will interact with the environment'''
    if A=='right':
        if S==N_STATES-2: ##terminate
            S_='terminal'
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def update_env(S,episode,step_counter):
    '''This is how environment be updated'''
    env_list=['-']*(N_STATES-1)+['T']  ##-----------T our environment
    if S=='terminal':
        interaction='Episode %s: totla_steps = %s'%(episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r              ',end='')
    else:
        env_list[S]='o'
        interaction=''.join(env_list)
        print('\r{}'.format(interaction),end='') ##\r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容
        time.sleep(FRESH_TIME)                    ##end=''  默认print打印完是换行的 如果写前面这句 这不换行


def rl():
    '''main part of RL loop'''
    q_table=bulid_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter=0
        S=0
        is_terminal=False
        update_env(S,episode,step_counter)
        while not is_terminal:
            A=choose_action(S,q_table)
            S_,R=get_env_feedback(S,A)  ##take action & get next state
            q_predict=q_table.ix[S,A]
            if S_!='terminal':
                q_target=R+LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target=R ##next state is terminal
                is_terminal=True
            q_table.ix[S,A]+=ALPHA*(q_target-q_predict)
            S=S_

            update_env(S,episode,step_counter+1)
            step_counter+=1
    return q_table

if __name__=='__main__':
    q_table=rl()
    print('Q-table')
    print(q_table)







