import numpy as np 
import matplotlib.pyplot as plt
import random
from lsst.ts.batoid.CloseLoopTask import CloseLoopTask
import argparse

import gym
from gym import Env, spaces

import time

class Telescope(Env):
    def __init__(self, name):
        super(Telescope, self).__init__()
        
        # Define observation space (elevation angle, fwhm)
        #self.observation_space = spaces.Box(low = 0.0 , 
        #                                    high = 4.0,
        #                                    dtype= np.float16) 
    
        # Define action space corresponding to all degres of freedom
        M2hex = {'low': -np.float16(np.ones(5)*1e-2),
                    'high': np.float16(np.ones(5)*1e-2)}

        Camhex = {'low': -np.float16(np.ones(5)*1e-2),
                    'high': np.float16(np.ones(5)*1e-2)}

        M2bend = {'low': -np.float16(np.ones(20)*1e-2),
                    'high': np.float16(np.ones(20)*1e-2)}

        M1M3bend = {'low': -np.float16(np.ones(20)*1e-2),
                    'high': np.float16(np.ones(20)*1e-2)}

        self.action_space = spaces.Box(low = np.concatenate(( M2hex['low'], 
                                                              Camhex['low'], 
                                                              M2bend['low'], 
                                                              M1M3bend['low'] )), 
                                        high = np.concatenate(( M2hex['high'], 
                                                              Camhex['high'], 
                                                              M2bend['high'], 
                                                              M1M3bend['high'] )), 
                                        dtype = np.float16)

        self.observation_space = spaces.Box(low =  np.float16(np.array([0,0])), 
                                        high =  np.float16(np.array([30,30])), 
                                        dtype = np.float16)

        # Set initial degree of freedom state
        self.dof_initial = np.zeros(50)

        # Initialize observation variables
        self.altitude = 27.0912
        self.obsId = 9006000

        # Define and initialize closed loop task
        self.closeLoopTask = CloseLoopTask()    
        args = argparse.Namespace(inst='comcam', filterType='ref', 
                        rotCam=0.0, m1m3FErr=0.05, 
                        numOfProc=1, iterNum=5, 
                        output=f'/home/guillemmh/aos/perturbations/{name}/', 
                        log_level=20, clobber=False, 
                        pipelineFile='', boresightDeg=[0.03, -0.02], 
                        skyFile='/home/guillemmh/aos/ts_batoid/tests/testData/sky/skyComCam.txt', eimage=None, 
                        zAngleInDeg= self.altitude)

        self.closeLoopTask.initializeRealLoop(args)
        self.closeLoopTask.batoidCmpt.applyLUT()
        

    def plot_reward_evolution(self):
        # Init the canvas 
        plt.plot(np.arange(len(self.rewards)), self.rewards)
        plt.tick_params(labelsize = 16)
        plt.ylabel('FWHM', fontsize = 20)
        plt.xlabel('Iteration', fontsize = 20)
        plt.draw()

    def optimal(self):
        gqEffFwhm, wfe, sensor_ids = self.closeLoopTask.fwhm_compute(self.ep_return, self.obsId, self.altitude, rotAngInDeg=0.0)

        self.closeLoopTask.ofcCalc.calculate_corrections(
                wfe=wfe,
                field_idx=sensor_ids,
                filter_name=str('R'),
                gain=-1,
                rot=0.0,
            )
        dofInUm = self.closeLoopTask.ofcCalc.ofc_controller.aggregated_state 

        return dofInUm

    def reset(self):
        # Reset the reward
        self.ep_return  = 0

        # Initialize observation variables
        self.altitude = 27.0912
        self.obsId = 9006000


        # Define and initialize closed loop task
        '''
        self.closeLoopTask = CloseLoopTask()    
        args = argparse.Namespace(inst='comcam', filterType='ref', 
                        rotCam=0.0, m1m3FErr=0.05, 
                        numOfProc=1, iterNum=5, 
                        output=f'/home/guillemmh/aos/perturbations/testing_ddpg/', 
                        log_level=20, clobber=False, 
                        pipelineFile='', boresightDeg=[0.03, -0.02], 
                        skyFile='/home/guillemmh/aos/ts_batoid/tests/testData/sky/skyComCam.txt', eimage=None, 
                        zAngleInDeg= self.altitude)

        self.closeLoopTask.initializeRealLoop(args)
        '''

        # Set Altitude
        self.closeLoopTask.batoidCmpt.setSurveyParam(
            obsId=self.obsId,
            zAngleInDeg=self.altitude,
            rotAngInDeg=0.0
        )

        self.closeLoopTask.batoidCmpt.setZAngle()
        self.closeLoopTask.batoidCmpt.applyLUT()

        # Obtain first observation
        self.closeLoopTask.batoidCmpt.setDofInUm(self.dof_initial)

        # Save the DOF file
        #self.closeLoopTask.batoidCmpt.saveDofInUmFileForNextIter(self.dof_initial)

        # Reward for executing a step.
        reward = self.closeLoopTask.fwhm_compute(self.ep_return, self.obsId, self.altitude, rotAngInDeg=0.0)

        # Draw elements on the canvas
        #self.plot_reward_evolution()

        # return the observation
        return (self.altitude, 0.0)


    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        action = np.float16(action)
        #assert self.action_space.contains(action), "Invalid Action"

        # Perform action
        self.closeLoopTask.batoidCmpt.setZAngle()
        self.closeLoopTask.batoidCmpt.applyLUT()
        self.closeLoopTask.batoidCmpt.setDofInUm(action)

        # Save the DOF file
        #self.closeLoopTask.batoidCmpt.saveDofInUmFileForNextIter(action)

        # Reward for executing a step.
        reward, _, _ = self.closeLoopTask.fwhm_compute(self.ep_return, self.obsId, self.altitude, rotAngInDeg=0.0)

        # Update next iteration observation
        self.obsId += 10
        #self.altitude = 27.0912

        dofInUm = self.closeLoopTask.batoidCmpt.getDofInUm()

        # Set Altitude
        #self.closeLoopTask.batoidCmpt.setSurveyParam(
        #    obsId=self.obsId,
        #    zAngleInDeg=self.altitude,
        #    rotAngInDeg=0.0
        #)

        #self.closeLoopTask.batoidCmpt.setZAngle()

        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        #self.plot_reward_evolution()

        return (self.altitude, 0.0), -reward, done, []
