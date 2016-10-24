#import math
import random
import argparse
from collections import defaultdict
import alpha_functions, epsilon_functions, gamma_functions
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize any additional variables here
        self.actions = Environment.valid_actions
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        
        # Report number of visits in a state
        self.negative_rewards = list()
        self.n_trial = 0
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.n_trial += 1
        
    def set_q_params(self, alpha, epsilon, gamma, default_q_value=0):
        # self.q_mapping = {}
        # Set default_q_value for 10 for optimistic learner
        self.q_mapping = defaultdict(lambda: default_q_value)
        self.alpha_function = alpha
        self.epsilon_function = epsilon
        self.gamma_function = gamma
         
    def get_q_state(self, inputs):
        '''Returns the state for the Q-learning algorithm.'''
        return (inputs['light'], inputs['left'], inputs['oncoming'], self.next_waypoint)
    
    def max_q_action(self):
        '''Returns the policy and the utility related to the smartcab state'''
        q_action = None
        q_value = None
        
        # shuffle actions so it can be a fair selection in case that all the actions have the same value
        random.shuffle(self.actions)
        
        for action in self.actions:
            if q_value is None or self.q_mapping[self.state, action] > q_value:
                q_action = action
                q_value = self.q_mapping[self.state, action]
        
        return q_action, q_value

    def update(self, t):
       # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # Initialize Q-learning parameters, using epsilon-greedy approach
        alpha = self.alpha_function(t)
        epsilon = self.epsilon_function(t)
        gamma = self.gamma_function(t)
        
        # Update state, where each state is represented by the tuple (left, oncoming, right, next_waypoint)
        self.state = self.get_q_state(inputs)
        
        # Select action according to your policy
        # Explore different actions with epsilon probabily
        if random.uniform(0, 1) <= epsilon:
            action = random.choice(self.actions)
            max_q_value = self.q_mapping[self.state, action]
        else:
            # Exploit the information learned so far using Q-function 
            action, max_q_value = self.max_q_action()

        # Execute action and get reward
        reward = self.env.act(self, action)
       
        # Learn policy based on state, action, reward
        if self.last_state is not None:
            self.q_mapping[self.last_state, self.last_action] = (
                (1 -alpha) * self.q_mapping[self.last_state, self.last_action] +
                alpha * (self.last_reward + gamma * max_q_value))
        
        # Record variables to use in the next iteration
        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward
        
        # Register negative rewards
        if reward < 0:
            while self.n_trial > len(self.negative_rewards):
                self.negative_rewards.append(list())
            self.negative_rewards[self.n_trial -1].append(reward)

       # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, next_waypoint: {}, time: {}".format(deadline, inputs, action, reward, self.next_waypoint, t)  # [debug]
        
    def report_q_learning(self, trials=100):
        # Print the last 'trials' negative rewards
        for n_trial, reward_list in enumerate(self.negative_rewards[-trials:]):
            print 'Trial {} - list: {}, length: {}, sum: {}'.format(len(self.negative_rewards) -trials + n_trial, reward_list, len(reward_list), sum(reward_list))

def run():
    """Run the agent for a finite number of trials."""
    
    # Make sure the results are reproducible
    random.seed(111)
    
    # Parse command line parameters
    parser = argparse.ArgumentParser(description='Smartcab Simulation')
    parser.add_argument('--no-deadline', action='store_true', default=False, dest='no_deadline',
                        help='Do not enforce deadline')
    parser.add_argument('--debug', action='store_true', default=False, dest='debug',
                        help='Run the simulation for all the combinations of (alpha, epsilon, gamma) available')
    parser.add_argument('--delay', action='store', default=0.5, dest='delay', type=float,
                        help='Delay of updating smartcab positions')
    parser.add_argument('--no-display', action='store_true', default=False, dest='no_display', 
                        help='Do not display the environment and the smartcabs')
    parser.add_argument('--n-trials', action='store', default=100, dest='trials', type=int,
                        help='Set number of trials of the simulation')
    args = parser.parse_args()
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=not args.no_deadline)  # specify agent to track with enforce_deadline=True by default
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    
    if not args.debug:
        # Set the best parameter functions
        a.set_q_params(
            alpha_functions.rational_alpha, 
            epsilon_functions.low_constant_epsilon, 
            gamma_functions.med_exponential_gamma)
        # Now simulate it
        sim = Simulator(e, update_delay=args.delay, display=not args.no_display)  # create simulator, uses pygame when display=True (default), if available
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=args.trials)  # run for a specified number of trials
        a.report_q_learning(trials=100)          # report q learning results
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    else:
        # Iterate through the alpha functions
        for alpha_name in dir(alpha_functions):
            # Skip what is not an alpha function
            if alpha_name.startswith('__') or not callable(getattr(alpha_functions, alpha_name)): 
                continue
            # Iterate through the epsilon functions
            for epsilon_name in dir(epsilon_functions):
                # Skip what is not an epsilon function
                if epsilon_name.startswith('__') or not callable(getattr(epsilon_functions, epsilon_name)): 
                    continue
                # Iterate through the gamma functions
                for gamma_name in dir(gamma_functions):
                    # Skip what is not an gamma function
                    if gamma_name.startswith('__') or not callable(getattr(gamma_functions, gamma_name)): 
                        continue
                        
                    print 'New simulation: alpha: {}, epsilon: {}, gamma: {}'.format(alpha_name, epsilon_name, gamma_name)
                    # Set functions (alpha, epsilon, gamma) to the underlying variables
                    a.set_q_params(
                        getattr(alpha_functions, alpha_name), 
                        getattr(epsilon_functions, epsilon_name),
                        getattr(gamma_functions, gamma_name))

                    # Now simulate it 
                    sim = Simulator(e, update_delay=args.delay, display=not args.no_display)  # create simulator, uses pygame when display=True (default), if available
                    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                    sim.run(n_trials=args.trials)  # run for a specified number of trials
                    a.report_q_learning()          # report q learning results
                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
