# Import routines

import numpy as np
import math
import random
from icecream import ic 


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [[l, h, day] for l in range(m) for h in range(t) for day in range(d)]
        self.state_init = random.choice(self.state_space)        
        # Start the first round
        self.reset()

    #lets define sum matrix assumptions
    
    def sta_loc(self, state):
        return int(state[0])

    def sta_time(self, state):
        return int(state[1])

    def sta_day(self, state):
        return int(state[2])

    def act_pickup(self, action):
        return action[0]

    def act_drop(self, action):
        return action[1]
    
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_size = m+t+d
        state_encod = [0 for i in range(state_size)]
        state_encod[self.sta_loc(state)] = 1
        state_encod[m+self.sta_time(state)] = 1
        state_encod[m+t+self.sta_day(state)] = 1
        
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)    

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        #actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        # lets fetch the loc and action
        
        # locations
        cur_loc = self.sta_loc(state)
        picup_loc = self.act_pickup(action)
        drop_loc = self.act_drop(action)
        
        #time and day
        cur_time = self.sta_time(state)
        cur_day = self.sta_day(state)
        
        #Initialising transit and waiting time
        transit_time = 0
        waiting_time = 0
        ride_time = 0
        if ((picup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            waiting_time = 1
        elif (cur_loc == picup_loc):
            ride_time = Time_matrix[cur_loc][drop_loc][cur_time][cur_day]
        else:
            ride_time = Time_matrix[picup_loc][drop_loc][cur_time][cur_day]
            transit_time = Time_matrix[cur_loc][picup_loc][cur_time][cur_day]

            
        total_trip_time = waiting_time + transit_time + ride_time
        
        reward = (R * ride_time) - (C * total_trip_time)
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        next_state = []
        # locations
        cur_loc = self.sta_loc(state)
        picup_loc = self.act_pickup(action)
        drop_loc = self.act_drop(action)
        next_loc = 0
        
        #time and day
        cur_time = self.sta_time(state)
        cur_day = self.sta_day(state)
        
        #Initialising transit and waiting time
        transit_time = 0
        waiting_time = 0
        ride_time = 0
        
        if ((picup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            waiting_time = 1
            t_new, d_new = self.calc_new_time_day(cur_time+1, cur_day)
            next_state=(cur_loc, t_new, d_new)
        elif (cur_loc == picup_loc):
            
            ride_time = Time_matrix[cur_loc][drop_loc][cur_time][cur_day]
            t_new,d_new = self.calc_new_time_day(ride_time, cur_day)
            next_state=(drop_loc, t_new, d_new)
            
        else:
            transit_time = Time_matrix[cur_loc][picup_loc][cur_time][cur_day]
            total_time = transit_time + cur_time
            t_new,d_new = self.calc_new_time_day(total_time, cur_day)
            ride_time = Time_matrix[picup_loc][drop_loc][t_new][d_new]
            t_new,d_new = self.calc_new_time_day(total_time, cur_day)
            next_state=(drop_loc, t_new, d_new)
        
        total_time_elapsed = waiting_time + transit_time + ride_time
 
        return next_state,total_time_elapsed


    ## Utility function

    def calc_new_time_day(self, time, day):
        if time >= 24:
            time = int(time - 24)
            if day == 6:
                day = 0
            else:
                day = int(day + 1)

        return int(time), day
    
    def reset(self):
        #return (list(range(0,10)))
        return self.action_space, self.state_space, self.state_init
