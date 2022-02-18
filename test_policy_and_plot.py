from mimetypes import init
import os
import torch
import numpy as np
import argparse
import pickle
import json
import time
import matplotlib.pyplot as plt

from cassie import CassieEnv

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2",help="path to directory that contains trained policy")

    args = parser.parse_args()

    # path
    dir_path = args.dir
    actor_path = os.path.join(dir_path, "actor.pt")
    parameters_path = os.path.join(dir_path, "experiment.pkl")
    eval_dir = os.path.join(dir_path, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    init_state_path = os.path.join(eval_dir,"init_full_state.json")
    json_path = os.path.join(eval_dir,"states_action.json")

    # load the run parameters
    parameters = pickle.load(open(parameters_path, "rb"))

    with open(init_state_path,"rb") as f:
        init_state = json.load(f)

    env_name = parameters.env_name
    trajectory_name = parameters.traj # Use this to choose the reference trajectory 
    dynamic_randomization = parameters.dyn_random # True or False
    no_delta = parameters.no_delta # True or False
    clock_based = parameters.clock_based # True or False
    state_est = parameters.state_est # True or False
    reward_function = parameters.reward # reward function
    history = parameters.history # history for FF nets
    
    print("env_name:",env_name)
    print("ref_traj:",trajectory_name)
    print("reward_function:",reward_function)
    print("dynamic random:",dynamic_randomization)
    print("no_delta:",no_delta)
    print("state_est:",state_est)
    print("clock_based:",clock_based)

    # load policy and initialize it
    policy = torch.load(actor_path)
    policy.eval()
    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    # construct environment

    if env_name == "Cassie-v0":
        env = CassieEnv(traj=trajectory_name,reward=reward_function, clock_based=clock_based, state_est=state_est, dynamics_randomization=dynamic_randomization, no_delta=no_delta, history=history)
    else:
        raise ValueError("Environment \"{}\" is not current supported".format(env_name))

    sim_state = env.render()

    action_state_list = []

    state,state_dict = env.initial_by_given_state(init_state)

    while sim_state:
        
        start_time = time.time()

        sim_state = env.render()

        end_time = time.time()
        
        if (not env.vis.ispaused()):

            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()

            action_state_list.append({"action":action.tolist(), "state_dict":state_dict, "state":state.tolist()})

            env.step(action)

            state, reward, done, state_dict = env.step(action)

        delaytime = max(0, 1000 / 30000 - (end_time-start_time))
        time.sleep(delaytime)
    
    with open(json_path,"w") as fn:
        json.dump(action_state_list, fn, indent=4)

    timestamp = [action_state["state_dict"]["timestamp"] for action_state in action_state_list]
    pelvis_height = [action_state["state_dict"]["robot_state"]["pelvis_height"] for action_state in action_state_list]

    plt.plot(timestamp,pelvis_height)
    plt.show()
    

if __name__ == "__main__":
    main()