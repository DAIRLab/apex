# Copyright (c) 2018 Dynamic Robotics Laboratory
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from .cassiemujoco_ctypes import *
import os
import ctypes
import numpy as np

# Get base directory
_dir_path = os.path.dirname(os.path.realpath(__file__))

# Initialize libcassiesim
# cassie_mujoco_init(str.encode(_dir_path+"/cassie_slosh_mass.xml"))
# cassie_mujoco_init(str.encode(_dir_path+"/cassie_yoke.xml"))
cassie_mujoco_init(str.encode(_dir_path+"/cassie.xml"))
# cassie_mujoco_init(str.encode("../model/cassie_hfield.xml"))


# Interface classes
# TODO: hard setting nbody and ngeom is not safe, much less safe than hard setting nv. If adding random geoms/bodies
# like of obstacles or just for visual purpose, will mess things up. Either make getting functions and get the values
# in the init or switch functions that nbody and ngeom to take in a name instead of letting you set all values at once
class CassieSim:
    def __init__(self, modelfile, reinit=False):
        self.c = cassie_sim_init(modelfile.encode('utf-8'), reinit)
        self.nv = 32
        self.nbody = 26
        self.nq = 35
        self.ngeom = 35

    def step(self, u):
        y = cassie_out_t()
        cassie_sim_step(self.c, y, u)
        return y

    def step_pd(self, u):
        y = state_out_t()
        cassie_sim_step_pd(self.c, y, u)
        return y

    def get_state(self):
        s = CassieState()
        cassie_get_state(self.c, s.s)
        return s

    def set_state(self, s):
        cassie_set_state(self.c, s.s)

    def time(self):
        timep = cassie_sim_time(self.c)
        return timep[0]

    def qpos(self):
        qposp = cassie_sim_qpos(self.c)
        return qposp[:35]

    def qvel(self):
        qvelp = cassie_sim_qvel(self.c)
        return qvelp[:32]

    def qacc(self):
        qaccp = cassie_sim_qacc(self.c)
        return qaccp[:32]

    def xquat(self, body_name):
        xquatp = cassie_sim_xquat(self.c, body_name.encode())
        return xquatp[:4]

    def set_time(self, time):
        timep = cassie_sim_time(self.c)
        timep[0] = time

    def set_qpos(self, qpos):
        qposp = cassie_sim_qpos(self.c)
        for i in range(min(len(qpos), 35)):
            qposp[i] = qpos[i]

    def set_qvel(self, qvel):
        qvelp = cassie_sim_qvel(self.c)
        for i in range(min(len(qvel), 32)):
            qvelp[i] = qvel[i]

    def set_qacc(self, qacc):
        qaccp = cassie_sim_qacc(self.c)
        for i in range(32):
            qaccp[i] = qacc[i]

    def hold(self):
        cassie_sim_hold(self.c)

    def release(self):
        cassie_sim_release(self.c)

    def apply_force(self, xfrc, body_name="cassie-pelvis"):
        xfrc_array = (ctypes.c_double * 6)()
        for i in range(len(xfrc)):
            xfrc_array[i] = xfrc[i]
        cassie_sim_apply_force(self.c, xfrc_array, body_name.encode())

    def foot_force(self, force):
        frc_array = (ctypes.c_double * 12)()
        cassie_sim_foot_forces(self.c, frc_array)
        for i in range(12):
            force[i] = frc_array[i]

    def foot_pos(self, pos):
        pos_array = (ctypes.c_double * 6)()
        cassie_sim_foot_positions(self.c, pos_array)
        for i in range(6):
            pos[i] = pos_array[i]

    def foot_vel(self, vel):
        vel_array = (ctypes.c_double * 12)()
        cassie_sim_foot_velocities(self.c, vel_array)
        for i in range(12):
            vel[i] = vel_array[i]

    def body_vel(self, vel, body_name):
        vel_array = (ctypes.c_double * 6)()
        cassie_sim_body_vel(self.c, vel_array, body_name.encode())
        for i in range(6):
            vel[i] = vel_array[i]

    def foot_quat(self, quat):
        quat_array = (ctypes.c_double * 4)()
        cassie_sim_foot_quat(self.c, quat_array)
        for i in range(4):
            quat[i] = quat_array[i]

    def clear_forces(self):
        cassie_sim_clear_forces(self.c)

    def get_foot_forces(self):
        y = state_out_t()
        force = np.zeros(12)
        self.foot_force(force)
        return force[[2, 8]]

    def get_dof_damping(self):
        ptr = cassie_sim_dof_damping(self.c)
        ret = np.zeros(self.nv)
        for i in range(self.nv):
          ret[i] = ptr[i]
        return ret
    
    def get_body_mass(self):
        ptr = cassie_sim_body_mass(self.c)
        ret = np.zeros(self.nbody)
        for i in range(self.nbody):
          ret[i] = ptr[i]
        return ret

    def get_body_ipos(self):
        nbody = self.nbody * 3
        ptr = cassie_sim_body_ipos(self.c)
        ret = np.zeros(nbody)
        for i in range(nbody):
          ret[i] = ptr[i]
        return ret

    def get_geom_friction(self):
        ptr = cassie_sim_geom_friction(self.c)
        ret = np.zeros(self.ngeom * 3)
        for i in range(self.ngeom * 3):
          ret[i] = ptr[i]
        return ret

    def get_geom_rgba(self):
        ptr = cassie_sim_geom_rgba(self.c)
        ret = np.zeros(self.ngeom * 4)
        for i in range(self.ngeom * 4):
          ret[i] = ptr[i]
        return ret

    def get_geom_quat(self):
        ptr = cassie_sim_geom_quat(self.c)
        ret = np.zeros(self.ngeom * 4)
        for i in range(self.ngeom * 4):
          ret[i] = ptr[i]
        return ret

    def set_dof_damping(self, data):
        c_arr = (ctypes.c_double * self.nv)()

        if len(data) != self.nv:
          print("SIZE MISMATCH SET_DOF_DAMPING()")
          exit(1)
        
        for i in range(self.nv):
          c_arr[i] = data[i]

        cassie_sim_set_dof_damping(self.c, c_arr)

    def set_body_mass(self, data, name=None):
        # If no name is provided, set ALL body masses and assume "data" is array
        # containing masses for every body
        if name is None:
            c_arr = (ctypes.c_double * self.nbody)()

            if len(data) != self.nbody:
                print("SIZE MISMATCH SET_BODY_MASS()")
                exit(1)
            
            for i in range(self.nbody):
                c_arr[i] = data[i]

            cassie_sim_set_body_mass(self.c, c_arr)
        # If name is provided, only set mass for specified body and assume
        # "data" is a single double
        else:
            cassie_sim_set_body_name_mass(self.c, name.encode(), ctypes.c_double(data))

    def set_body_ipos(self, data):
        nbody = self.nbody * 3
        c_arr = (ctypes.c_double * nbody)()

        if len(data) != nbody:
          print("SIZE MISMATCH SET_BODY_IPOS()")
          exit(1)
        
        for i in range(nbody):
          c_arr[i] = data[i]

        cassie_sim_set_body_ipos(self.c, c_arr)

    def set_geom_friction(self, data, name=None):
        if name is None:
            c_arr = (ctypes.c_double * (self.ngeom*3))()

            if len(data) != self.ngeom*3:
                print("SIZE MISMATCH SET_GEOM_FRICTION()")
                exit(1)

            for i in range(self.ngeom*3):
                c_arr[i] = data[i]

            cassie_sim_set_geom_friction(self.c, c_arr)
        else:
            fric_array = (ctypes.c_double * 3)()
            for i in range(3):
                fric_array[i] = data[i]
            cassie_sim_set_geom_name_friction(self.c, name.encode(), fric_array)


    def set_geom_rgba(self, data):
        ngeom = self.ngeom * 4

        if len(data) != ngeom:
            print("SIZE MISMATCH SET_GEOM_RGBA()")
            exit(1)

        c_arr = (ctypes.c_float * ngeom)()

        for i in range(ngeom):
            c_arr[i] = data[i]

        cassie_sim_set_geom_rgba(self.c, c_arr)
    
    def set_geom_quat(self, data, name=None):
        if name is None:
            ngeom = self.ngeom * 4

            if len(data) != ngeom:
                print("SIZE MISMATCH SET_GEOM_QUAT()")
                exit(1)

            c_arr = (ctypes.c_double * ngeom)()

            for i in range(ngeom):
                c_arr[i] = data[i]

            cassie_sim_set_geom_quat(self.c, c_arr)
        else:
            quat_array = (ctypes.c_double * 4)()
            for i in range(4):
                quat_array[i] = data[i]
            cassie_sim_set_geom_name_quat(self.c, name.encode(), quat_array)

    
    def set_const(self):
        cassie_sim_set_const(self.c)

    def full_reset(self):
        cassie_sim_full_reset(self.c)

    def get_hfield_nrow(self):
        return cassie_sim_get_hfield_nrow(self.c)

    def get_hfield_ncol(self):
        return cassie_sim_get_hfield_ncol(self.c)

    def get_nhfielddata(self):
        return cassie_sim_get_nhfielddata(self.c)

    def get_hfield_size(self):
        ret = np.zeros(4)
        ptr = cassie_sim_get_hfield_size(self.c)
        for i in range(4):
            ret[i] = ptr[i]
        return ret

    # Note that data has to be a flattened array. If flattening 2d numpy array, rows are y axis
    # and cols are x axis. The data must also be normalized to (0-1)
    def set_hfield_data(self, data):
        nhfielddata = self.get_nhfielddata()
        if len(data) != nhfielddata:
            print("SIZE MISMATCH SET_HFIELD_DATA")
            exit(1)
        data_arr = (ctypes.c_float * nhfielddata)(*data)
        cassie_sim_set_hfielddata(self.c, ctypes.cast(data_arr, ctypes.POINTER(ctypes.c_float)))
    
    def get_hfield_data(self):
        nhfielddata = self.get_nhfielddata()
        ret = np.zeros(nhfielddata)
        ptr = cassie_sim_hfielddata(self.c)
        for i in range(nhfielddata):
            ret[i] = ptr[i]
        return ret

    def set_hfield_size(self, data):
        if len(data) != 4:
            print("SIZE MISMATCH SET_HFIELD_SIZE")
            exit(1)
        size_array = (ctypes.c_double * 4)()
        for i in range(4):
            size_array[i] = data[i]
        cassie_sim_set_hfield_size(self.c, size_array)

    def copy(self, src):
        cassie_sim_copy(self.c, src.c)

    def copy_just_sim(self, src):
        cassie_sim_copy_just_sim(self.c, src.c)

    def duplicate(self):
        return cassie_sim_duplicate(self.c)

    def copy_mjd(self, src):
        cassie_sim_copy_mjd(self.c, src.c)

    def copy_state_est(self, src):
        # print("in python copy")
        cassie_sim_copy_state_est(self.c, src.c)
        # print("after copy")

    def get_cassie_out(self):
        return cassie_sim_get_cassie_out(self.c)

    def run_state_est(self, cassie_out):
        y = state_out_t()
        cassie_sim_run_state_est(self.c, cassie_out, y)
        return y

    def get_act_vel(self):
        act_vel_p = cassie_sim_act_vel(self.c)
        return act_vel_p[:10]

    def set_act_vel(self, act_vel):
        act_vel_p = cassie_sim_act_vel(self.c)
        for i in range(10):
            act_vel_p[i] = act_vel[i]

    def get_sensordata(self):
        sensor_p = cassie_sim_sensordata(self.c)
        return sensor_p[:29]

    def set_sensordata(self, sdata):
        sensor_p = cassie_sim_sensordata(self.c)
        for i in range(29):
            sensor_p[i] = sdata[i]

    # Returns a pointer to an array of joint_filter_t objects. Can be accessed/indexed as a usual python array of
    # joint filter objects
    def get_joint_filter(self):
        j_filters = cassie_sim_joint_filter(self.c)
        out_filters = (joint_filter_t*6)()
        for i in range(6):
            for j in range(4):
                out_filters[i].x[j] = j_filters[i].x[j]
            for j in range(3):
                out_filters[i].y[j] = j_filters[i].y[j]
        return out_filters
        # return j_filters

    # Set interal state of the joint filters. Takes in an c struct array object of joint_filter_t objects,
    # which can be initialized like "j=(joint_filter_t*size)()" and then accessed as a usual python list. 
    # See cassiemujoco_ctypes for definition of joint_filter_t object
    def set_joint_filter(self, joint_filters):
        cassie_sim_set_joint_filter(self.c, joint_filters)

    def set_joint_filter2(self, x, y):
        x_arr = (ctypes.c_double * (6*4))(*x)
        y_arr = (ctypes.c_double * (6*3))(*y)
        cassie_sim_set_joint_filter2(self.c, ctypes.cast(x_arr, ctypes.POINTER(ctypes.c_double)), ctypes.cast(y_arr, ctypes.POINTER(ctypes.c_double)))

    # Returns a pointer to an array of drive_filter_t objects. Can be accessed/indexed as a usual python array of
    # drive filter objects
    def get_drive_filter(self):
        d_filters = cassie_sim_drive_filter(self.c)
        out_filters = (drive_filter_t*10)()
        for i in range(10):
            for j in range(9):
                out_filters[i].x[j] = d_filters[i].x[j]
        return out_filters

    # Set interal state of the drive filters. Takes in an c struct array object of drive_filter_t objects,
    # which can be initialized like "j=(drive_filter_t*size)()" and then accessed as a usual python list. 
    # See cassiemujoco_ctypes for definition of drive_filter_t object
    def set_drive_filter(self, drive_filters):
        cassie_sim_set_drive_filter(self.c, drive_filters)

    def set_drive_filter2(self, x):
        x_arr = (ctypes.c_int * (10*9))(*x)
        cassie_sim_set_drive_filter2(self.c, ctypes.cast(x_arr, ctypes.POINTER(ctypes.c_int)))

    # Get the current state of the torque delay array. Returns a 2d numpy array of size (10, 6), 
    # number of motors by number of delay cycles
    def get_torque_delay(self):
        t_arr = (ctypes.c_double * 60)()
        cassie_sim_torque_delay(self.c, t_arr)
        # return np.array(t_arr[:]).reshape((10, 6))
        return np.array(t_arr[:])

    # Set the torque delay state. Takes in a 2d numpy array of size (10, 6), number of motors by number of delay cycles
    def set_torque_delay(self, data):
        set_t_arr = (ctypes.c_double * 60)(*data)
        cassie_sim_set_torque_delay(self.c, ctypes.cast(set_t_arr, ctypes.POINTER(ctypes.c_double)))

    def __del__(self):
        cassie_sim_free(self.c)

class CassieVis:
    def __init__(self, c, modelfile):
        self.v = cassie_vis_init(c.c, modelfile.encode('utf-8'))

    def draw(self, c):
        state = cassie_vis_draw(self.v, c.c)
        return state

    def valid(self):
        return cassie_vis_valid(self.v)

    def ispaused(self):
        return cassie_vis_paused(self.v)

    # Applies the inputted force to the inputted body. "xfrc_apply" should contain the force/torque to 
    # apply in Cartesian coords as a 6-long array (first 3 are force, last 3 are torque). "body_name" 
    # should be a string matching a body name in the XML file. If "body_name" doesn't match an existing
    # body name, then no force will be applied. 
    def apply_force(self, xfrc_apply, body_name):
        xfrc_array = (ctypes.c_double * 6)()
        for i in range(len(xfrc_apply)):
            xfrc_array[i] = xfrc_apply[i]
        cassie_vis_apply_force(self.v, xfrc_array, body_name.encode())

    def reset(self):
        cassie_vis_full_reset(self.v)

    def set_cam(self, body_name, zoom, azimuth, elevation):
        cassie_vis_set_cam(self.v, body_name.encode(), zoom, azimuth, elevation)

    def __del__(self):
        cassie_vis_free(self.v)

class CassieState:
    def __init__(self):
        self.s = cassie_state_alloc()

    def time(self):
        timep = cassie_state_time(self.s)
        return timep[0]

    def qpos(self):
        qposp = cassie_state_qpos(self.s)
        return qposp[:35]

    def qvel(self):
        qvelp = cassie_state_qvel(self.s)
        return qvelp[:32]

    def set_time(self, time):
        timep = cassie_state_time(self.s)
        timep[0] = time

    def set_qpos(self, qpos):
        qposp = cassie_state_qpos(self.s)
        for i in range(min(len(qpos), 35)):
            qposp[i] = qpos[i]

    def set_qvel(self, qvel):
        qvelp = cassie_state_qvel(self.s)
        for i in range(min(len(qvel), 32)):
            qvelp[i] = qvel[i]

    def __del__(self):
        cassie_state_free(self.s)

class CassieUdp:
    def __init__(self, remote_addr='127.0.0.1', remote_port='25000',
                 local_addr='0.0.0.0', local_port='25001'):
        self.sock = udp_init_client(str.encode(remote_addr),
                                    str.encode(remote_port),
                                    str.encode(local_addr),
                                    str.encode(local_port))
        self.packet_header_info = packet_header_info_t()
        self.recvlen = 2 + 697
        self.sendlen = 2 + 58
        self.recvlen_pd = 2 + 493
        self.sendlen_pd = 2 + 476
        self.recvbuf = (ctypes.c_ubyte * max(self.recvlen, self.recvlen_pd))()
        self.sendbuf = (ctypes.c_ubyte * max(self.sendlen, self.sendlen_pd))()
        self.inbuf = ctypes.cast(ctypes.byref(self.recvbuf, 2),
                                 ctypes.POINTER(ctypes.c_ubyte))
        self.outbuf = ctypes.cast(ctypes.byref(self.sendbuf, 2),
                                  ctypes.POINTER(ctypes.c_ubyte))

    def send(self, u):
        pack_cassie_user_in_t(u, self.outbuf)
        send_packet(self.sock, self.sendbuf, self.sendlen, None, 0)

    def send_pd(self, u):
        pack_pd_in_t(u, self.outbuf)
        send_packet(self.sock, self.sendbuf, self.sendlen_pd, None, 0)

    def recv_wait(self):
        nbytes = -1
        while nbytes != self.recvlen:
            nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen,
                                       None, None)
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        cassie_out = cassie_out_t()
        unpack_cassie_out_t(self.inbuf, cassie_out)
        return cassie_out

    def recv_wait_pd(self):
        nbytes = -1
        while nbytes != self.recvlen_pd:
            nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen_pd,
                                       None, None)
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        state_out = state_out_t()
        unpack_state_out_t(self.inbuf, state_out)
        return state_out

    def recv_newest(self):
        nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen,
                                   None, None)
        if nbytes != self.recvlen:
            return None
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        cassie_out = cassie_out_t()
        unpack_cassie_out_t(self.inbuf, cassie_out)
        return cassie_out

    def recv_newest_pd(self):
        nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen_pd,
                                   None, None)
        if nbytes != self.recvlen_pd:
            return None
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        state_out = state_out_t()
        unpack_state_out_t(self.inbuf, state_out)
        return state_out

    def delay(self):
        return ord(self.packet_header_info.delay)

    def seq_num_in_diff(self):
        return ord(self.packet_header_info.seq_num_in_diff)

    def __del__(self):
        udp_close(self.sock)
