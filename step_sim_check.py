import pickle
from matplotlib import pyplot as plt
import numpy as np
import cassie
import time
from tempfile import TemporaryFile
import os, re
import argparse
import torch
from util import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim
from cassie.quaternion_function import *

@torch.no_grad()
def process_data(policy, env_fn, num_steps):

    real_sim = env_fn() # The "real" sim, what we will consider ground truth
    outside_sim = env_fn() # The sim that will call step_pd outside of the env object
    control_sim = env_fn() # The "control" sim, which will call step_sim_basic like real_sim

    # Reset for test all 3 envs
    real_sim.reset_for_test()
    outside_sim.reset_for_test()
    control_sim.reset_for_test()
    real_sim.speed = 0
    real_sim.phase_add = 1
    control_sim.speed = 0
    control_sim.phase_add = 1
    outside_sim.speed = 0
    outside_sim.phase_add = 1

    for i in range(num_steps):
        print("STEP {}".format(i))
        # Get action from policy
        real_state = real_sim.get_full_state()
        control_state = control_sim.get_full_state()
        outside_state = outside_sim.get_full_state()
        action = policy.forward(torch.Tensor(real_state), deterministic=True).detach().numpy()
        control_action = policy.forward(torch.Tensor(control_state), deterministic=True).detach().numpy()
        outside_action = policy.forward(torch.Tensor(outside_state), deterministic=True).detach().numpy()
        # Double check that the input state and action are the same for all envs
        print("Before state diff, outside: {}\tcontrol: {}".format(np.linalg.norm(real_state-outside_state), np.linalg.norm(real_state-control_state)))
        print("action diff, outside: {}\tcontrol: {}".format(np.linalg.norm(action-outside_action), np.linalg.norm(action-control_action)))
        # Pass in action to step_sim_basic for real_sim and control_sim
        for j in range(real_sim.simrate):
            real_sim.step_sim_basic(action)
            control_sim.step_sim_basic(control_action)

            # For outside_sim manually make control u and call step_pd in this function. Use P, D and offset from real_sim 
            # so code will be identical to step_sim_basic code. The below code is straight up copied from the "step_sim_basic" function in 
            # cassie.py, just replace all "self" with "outside_sim"
            if not outside_sim.ik_baseline:
                if outside_sim.aslip_traj and outside_sim.phase == outside_sim.phaselen - 1:
                    ref_pos, ref_vel = outside_sim.get_ref_state(0)
                else:
                    ref_pos, ref_vel = outside_sim.get_ref_state(outside_sim.phase + outside_sim.phase_add)
            else:
                ref_pos = outside_sim.trajectory.ik_pos[outside_sim.simsteps]

            if not outside_sim.no_delta:
                outside_sim.offset = ref_pos[outside_sim.pos_idx]
            target = action + outside_sim.offset

            if outside_sim.joint_rand:
                target -= outside_sim.motor_encoder_noise
            outside_sim.u = pd_in_t()
            for k in range(5):
                outside_sim.u.leftLeg.motorPd.pGain[k]  = outside_sim.P[k]
                outside_sim.u.rightLeg.motorPd.pGain[k] = outside_sim.P[k]

                outside_sim.u.leftLeg.motorPd.dGain[k]  = outside_sim.D[k]
                outside_sim.u.rightLeg.motorPd.dGain[k] = outside_sim.D[k]

                outside_sim.u.leftLeg.motorPd.torque[k]  = 0 # Feedforward torque
                outside_sim.u.rightLeg.motorPd.torque[k] = 0 

                outside_sim.u.leftLeg.motorPd.pTarget[k]  = outside_action[k] + outside_sim.offset[k]
                outside_sim.u.rightLeg.motorPd.pTarget[k] = outside_action[k + 5] + outside_sim.offset[k + 5]

                outside_sim.u.leftLeg.motorPd.dTarget[k]  = 0
                outside_sim.u.rightLeg.motorPd.dTarget[k] = 0

            # Can uncomment this to absolutely double check that passing in the same control (we are, diff is 0). Should be zero
            # any way since resulting qpos is the same between all envs. 
            # u_diff = np.linalg.norm(np.array(mj_u.leftLeg.motorPd.pGain[:]) - np.array(real_sim.u.leftLeg.motorPd.pGain[:])) + \
            #         np.linalg.norm(np.array(mj_u.leftLeg.motorPd.dGain[:]) - np.array(real_sim.u.leftLeg.motorPd.dGain[:])) + \
            #         np.linalg.norm(np.array(mj_u.leftLeg.motorPd.torque[:]) - np.array(real_sim.u.leftLeg.motorPd.torque[:])) + \
            #         np.linalg.norm(np.array(mj_u.leftLeg.motorPd.pTarget[:]) - np.array(real_sim.u.leftLeg.motorPd.pTarget[:])) + \
            #         np.linalg.norm(np.array(mj_u.leftLeg.motorPd.dTarget[:]) - np.array(real_sim.u.leftLeg.motorPd.dTarget[:])) + \
            #         np.linalg.norm(np.array(mj_u.rightLeg.motorPd.pGain[:]) - np.array(real_sim.u.rightLeg.motorPd.pGain[:])) + \
            #         np.linalg.norm(np.array(mj_u.rightLeg.motorPd.dGain[:]) - np.array(real_sim.u.rightLeg.motorPd.dGain[:])) + \
            #         np.linalg.norm(np.array(mj_u.rightLeg.motorPd.torque[:]) - np.array(real_sim.u.rightLeg.motorPd.torque[:])) + \
            #         np.linalg.norm(np.array(mj_u.rightLeg.motorPd.pTarget[:]) - np.array(real_sim.u.rightLeg.motorPd.pTarget[:])) + \
            #         np.linalg.norm(np.array(mj_u.rightLeg.motorPd.dTarget[:]) - np.array(real_sim.u.rightLeg.motorPd.dTarget[:]))
            # print("udiff: ", u_diff)

            outside_sim.cassie_state = outside_sim.sim.step_pd(outside_sim.u)
            # Can double check that step_sim_basic works fine. If use the below line instead of the line above, get expected behavior
            # outside_sim.step_sim_basic(outside_action)

            # Calculate all the differences in the resulting state estimator state
            pel_pos = np.array(real_sim.cassie_state.pelvis.position[:]) - np.array(outside_sim.cassie_state.pelvis.position[:])
            pel_orient = np.array(real_sim.cassie_state.pelvis.orientation[:]) - np.array(outside_sim.cassie_state.pelvis.orientation[:])
            pel_trans_vel = np.array(real_sim.cassie_state.pelvis.translationalVelocity[:]) - np.array(outside_sim.cassie_state.pelvis.translationalVelocity[:])
            pel_rot_vel = np.array(real_sim.cassie_state.pelvis.rotationalVelocity[:]) - np.array(outside_sim.cassie_state.pelvis.rotationalVelocity[:])
            pel_accel = np.array(real_sim.cassie_state.pelvis.translationalAcceleration[:]) - np.array(outside_sim.cassie_state.pelvis.translationalAcceleration[:])

            motors_pos = np.array(real_sim.cassie_state.motor.position[:]) - np.array(outside_sim.cassie_state.motor.position[:])
            motors_vel = np.array(real_sim.cassie_state.motor.velocity[:]) - np.array(outside_sim.cassie_state.motor.velocity[:])
            joints_pos = np.array(real_sim.cassie_state.joint.position[:]) - np.array(outside_sim.cassie_state.joint.position[:])
            joints_vel = np.array(real_sim.cassie_state.joint.velocity[:]) - np.array(outside_sim.cassie_state.joint.velocity[:])
            torques = np.array(real_sim.cassie_state.motor.torque[:]) - np.array(outside_sim.cassie_state.motor.torque[:])

            l_foot_pos = np.array(real_sim.cassie_state.leftFoot.position[:]) - np.array(outside_sim.cassie_state.leftFoot.position[:])
            l_foot_orient = np.array(real_sim.cassie_state.leftFoot.orientation[:]) - np.array(outside_sim.cassie_state.leftFoot.orientation[:])
            l_foot_trans_vel = np.array(real_sim.cassie_state.leftFoot.footTranslationalVelocity[:]) - np.array(outside_sim.cassie_state.leftFoot.footTranslationalVelocity[:])
            l_foot_rot_vel = np.array(real_sim.cassie_state.leftFoot.footRotationalVelocity[:]) - np.array(outside_sim.cassie_state.leftFoot.footRotationalVelocity[:])
            l_foot_toe_force = np.array(real_sim.cassie_state.leftFoot.toeForce[:]) - np.array(outside_sim.cassie_state.leftFoot.toeForce[:])
            l_foot_heel_force = np.array(real_sim.cassie_state.leftFoot.heelForce[:]) - np.array(outside_sim.cassie_state.leftFoot.heelForce[:])
            
            r_foot_pos = np.array(real_sim.cassie_state.rightFoot.position[:]) - np.array(outside_sim.cassie_state.rightFoot.position[:])
            r_foot_orient = np.array(real_sim.cassie_state.rightFoot.orientation[:]) - np.array(outside_sim.cassie_state.rightFoot.orientation[:])
            r_foot_trans_vel = np.array(real_sim.cassie_state.rightFoot.footTranslationalVelocity[:]) - np.array(outside_sim.cassie_state.rightFoot.footTranslationalVelocity[:])
            r_foot_rot_vel = np.array(real_sim.cassie_state.rightFoot.footRotationalVelocity[:]) - np.array(outside_sim.cassie_state.rightFoot.footRotationalVelocity[:])
            r_foot_toe_force = np.array(real_sim.cassie_state.rightFoot.toeForce[:]) - np.array(outside_sim.cassie_state.rightFoot.toeForce[:])
            r_foot_heel_force = np.array(real_sim.cassie_state.rightFoot.heelForce[:]) - np.array(outside_sim.cassie_state.rightFoot.heelForce[:])

            real_state = real_sim.get_full_state()
            control_state = control_sim.get_full_state()
            outside_state = outside_sim.get_full_state()
            ### Can uncomment to see that the state estimator immediately differs after calling step_pd
            # print(np.max(real_state-outside_state), np.argmax(real_state-outside_state))
            # print("{} state diff, outside: {}\tcontrol: {}".format(j, np.linalg.norm(real_state-outside_state), np.linalg.norm(real_state-control_state)))

        # Compare the resulting qpos
        real_qpos = np.array(real_sim.sim.qpos())
        outside_qpos = np.array(outside_sim.sim.qpos())
        control_qpos = np.array(control_sim.sim.qpos())
        outside_diff = np.linalg.norm(real_qpos - outside_qpos)
        control_diff = np.linalg.norm(real_qpos - control_qpos)

        # The qpos are all the same after the first step because all envs took the same control step. However, somehow state est differs
        # For the 2nd step, qpos is different because state est difference made the envs took different actions. 
        print("qpos diff, outside: {}\tcontrol: {}".format(outside_diff, control_diff))

        # Can see that the input state is different because of different state estimator outputs
        real_state = real_sim.get_full_state()
        control_state = control_sim.get_full_state()
        outside_state = outside_sim.get_full_state()
        print("After state diff, outside: {}\tcontrol: {}".format(np.linalg.norm(real_state-outside_state), np.linalg.norm(real_state-control_state)) )

        # Increment the phases for all envs
        real_sim.phase += 1
        control_sim.phase += 1
        outside_sim.phase += 1
        if real_sim.phase >= real_sim.phaselen:
            real_sim.phase -= real_sim.phaselen
            control_sim.phase -= control_sim.phaselen
            outside_sim.phase -= outside_sim.phaselen

pol_path = "./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2/"

run_args = pickle.load(open(pol_path + "experiment.pkl", "rb"))

if not hasattr(run_args, "simrate"):
    run_args.simrate = 50
    print("manually choosing simrate as 50 (40 Hz)")
if not hasattr(run_args, "phase_based"):
    run_args.phase_based = False

policy = torch.load(pol_path + "actor.pt")
policy.eval()
run_args.learn_gains = False
run_args.ik_baseline = False

env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, phase_based=run_args.phase_based, clock_based=run_args.clock_based, state_est=run_args.state_est, no_delta=run_args.no_delta, learn_gains=run_args.learn_gains, ik_baseline=run_args.ik_baseline, dynamics_randomization=run_args.dyn_random, mirror=run_args.mirror, reward=run_args.reward, history=run_args.history)
env = env_fn()

process_data(policy, env_fn, 2)