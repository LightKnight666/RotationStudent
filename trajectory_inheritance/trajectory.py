# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
import numpy as np
from os import path
import os
import pickle
from Directories import directory
from copy import copy

""" Making Directory Structure """
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['Small Far', 'Small Near', 'Medium', 'Large'],
         'humanhand': ''}

solvers = ['ant', 'human', 'humanhand', 'ps_simulation']

length_unit = {'ant': 'cm', 'human': 'm', 'humanhand': 'cm', 'ps_simulation': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


class Trajectory:
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, VideoChain=None):
        self.shape = shape  # shape (maybe this will become name of the maze...) (H, I, T, SPT)
        self.size = size  # size (XL, SL, L, M, S, XS)
        self.solver = solver  # ant, human, sim, humanhand
        self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        if VideoChain is None:
            self.VideoChain = [self.filename]
        else:
            self.VideoChain = VideoChain
        self.fps = fps  # frames per second
        self.position = np.empty((1, 2), float)  # np.array of x and y positions of the centroid of the shape
        self.angle = np.empty((1, 1), float)  # np.array of angles while the shape is moving
        self.frames = np.empty(0, float)
        self.winner = winner  # whether the shape crossed the exit
        self.participants = None

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def step(self, my_maze, i, display=None):
        my_maze.set_configuration(self.position[i], self.angle[i])

    def divide_into_parts(self) -> list:
        """
        In order to treat the connections different than the actually tracked part, this function will split a single
        trajectory object into multiple trajectory objects.
        :return:
        """
        frame_dividers = [-1] + \
                         [i for i, (f1, f2) in enumerate(zip(self.frames, self.frames[1:])) if not f1 == f2 - 1] + \
                         [len(self.frames)]

        if len(frame_dividers) - 1 != len(self.VideoChain):
            raise Exception('Why are your frames not matching your VideoChain in ' + self.filename + ' ?')

        parts = [Trajectory_part(self, [chain_element], [fr1 + 1, fr2 + 1])
                 for chain_element, fr1, fr2 in zip(self.VideoChain, frame_dividers, frame_dividers[1:])]
        return parts

    def timer(self):
        """

        :return: time in seconds
        """
        return (len(self.frames) - 1) / self.fps

    def iterate_coords(self, step=1) -> iter:
        """
        Iterator over (x, y, theta) of the trajectory
        :return: tuple (x, y, theta) of the trajectory
        """
        for pos, angle in zip(self.position[::step, :], self.angle[::step]):
            yield pos[0], pos[1], angle

    def has_forcemeter(self):
        return False

    def old_filenames(self, i: int):
        if i > 0:
            raise Exception('only one old filename available')
        return self.filename

    def check(self) -> None:
        """
        Simple check, whether the object makes sense. It would be better to create a setter function, that ensures, that
        all the attributes make sense...
        """
        if self.frames.shape != self.angle.shape:
            raise Exception('Your frame shape does not match your angle shape!')

    def cut_off(self, frames: list):
        """

        :param frames: frame indices (not the yellow numbers on top)
        :return:
        """
        new = copy(self)
        new.frames = self.frames[frames[0]:frames[1]]
        new.position = self.position[frames[0]:frames[1]]
        new.angle = self.angle[frames[0]:frames[1]]
        return new

    def easy_interpolate(self, frames_list: list):
        """

        :param frames_list: list of lists of frame indices (not the yellow numbers on top)
        :return:
        """
        new = copy(self)
        for frames in frames_list:
            new.position[frames[0]:frames[1]] = np.vstack(
                [new.position[frames[0]] for _ in range(frames[1] - frames[0])])
            new.angle[frames[0]:frames[1]] = np.hstack([new.angle[frames[0]] for _ in range(frames[1] - frames[0])])
        return new

    def geometry(self):
        pass

    def initial_cond(self):
        """
        We changed the initial condition. First, we had the SPT start between the two slits.
        Later we made it start in the back of the room.
        :return: str 'back' or 'front' depending on where the shape started
        """
        if self.shape != 'SPT':
            return None
        elif self.position[0, 0] < Maze(self).slits[0]:
            return 'back'
        return 'front'

    def communication(self):
        return False


class Trajectory_part(Trajectory):
    def __init__(self, parent_traj, VideoChain: list, frames: list):
        """

        :param parent_traj: trajectory that the part is taken from
        :param VideoChain: list of names of videos that are supposed to be part of the trajectory part
        :param frames: []
        """
        super().__init__(size=parent_traj.size, shape=parent_traj.shape, solver=parent_traj.solver,
                         filename=parent_traj.filename, fps=parent_traj.fps, winner=parent_traj.winner,
                         VideoChain=VideoChain)
        self.parent_traj = parent_traj
        self.frames_of_parent = frames
        self.frames = parent_traj.frames[frames[0]:frames[-1]]
        self.position = parent_traj.position[frames[0]:frames[-1]]
        self.angle = parent_traj.angle[frames[0]:frames[-1]]

    def is_connector(self):
        return 'CONNECTOR' in self.VideoChain[-1]

    def geometry(self):
        return self.parent_traj.geometry()


def get(filename) -> Trajectory:
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be unpickled
    :return: trajectory object
    """
    # this is local on your computer
    if filename in os.listdir(directory):
        with open(path.join(directory, filename), 'rb') as f:
            print('You are loading ' + filename + 'from local copy.')
            x = Trajectory()
            (x.shape, x.size, x.solver, x.filename, x.fps, x.position, x.angle, x.frames, x.winner) = pickle.load(f)
        return x
    else:
        raise ValueError('I cannot find ' + filename)
