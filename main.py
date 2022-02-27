from trajectory_inheritance.trajectory import get, Trajectory
from Analysis.PathLength import PathLength
from matplotlib import pyplot as plt
from Directories import directory
import os
import numpy as np


def smooth_v(v) -> np.array:
    """
    Smooth the v using np.medfilt (smoothing window ~30)
    :param v: velocity
    :return: smoothed velocity
    """

    return


def v_max(v) -> float:
    """
    :param v: velocity
    :return: maximal velocity
    """

    return


def acceleration_frames(v, v_max) -> tuple:
    """

    """
    f1 = 0
    f2 = 100
    return f1, f2


# how long does it take for the shape to reach from 0 to v_max or from v_max to 0? Better: How much path is traversed.


def find_acceleration_frames(x: Trajectory) -> list:
    # TODO Rotation student: Find for a given trajectory, example frames, where the load is accelerated from 0 to v_max,
    #   or where the shape is decelerated (far away from the wall)

    # Find frames, where v is about v_max

    # Find frames beforehand, where the speed is 0.

    pass


def get_experiments() -> dict:
    exps = {'Large': [], 'Medium': [], 'Small Far': []}
    for file in os.listdir(directory):
        if file.startswith('l'):
            exps['Large'].append(file)
        elif file.startswith('m'):
            exps['Medium'].append(file)
        elif file.startswith('s'):
            exps['Small Far'].append(file)
    return exps


if __name__ == '__main__':
    path_list = list()  # list
    shape, solver = 'SPT', 'human'
    geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')  # tuple
    resolution_dict = dict()
    exp = get_experiments()
    for size in exp.keys():
        for name in exp[size]:
            trajectory = get(name)

            # Variables you can access
            # trajectory.position
            # trajectory.angle
            # trajectory.fps gives frame rate

            # TODO: Plot position (x, t), plot (v, t) (-> matplotlib.pyplot), v = np.sqrt((v_x**2 + v_y**2)

            # TODO: smooth_v, plot v

            # TODO: Write function v_max()

            # TODO: Write function find_acceleration_frames
            frames = find_acceleration_frames(trajectory)

            path_list.append(PathLength(trajectory).calculate_path_length(frames=frames))

        # resolution_dict[size] = resolution(size, shape, solver, geometry, np.mean(path_list))
        resolution_dict[size] = np.mean(path_list)
    print(resolution_dict)
