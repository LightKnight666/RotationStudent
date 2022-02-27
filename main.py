from trajectory_inheritance.trajectory import get, Trajectory
from Analysis.PathLength import PathLength
from matplotlib import pyplot as plt
from Directories import directory
import os
import numpy as np


def velocity(position: np.array, fps: int) -> np.array:
    v = np.array([])
    return v


def smooth_v(v) -> np.array:
    """
    Smooth the v using np.medfilt (smoothing window ~30)
    :param v: velocity
    :return: smoothed velocity
    """

    return np.array([])


def calculate_v_max(v) -> float:
    """
    :param v: velocity
    :return: maximal velocity
    """

    return float()


def acceleration_frames(v, v_max) -> list:
    """

    """
    f1 = 0
    f2 = 100
    return [f1, f2]


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
    shape, solver = 'SPT', 'human'
    geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')  # tuple
    resolution_dict = dict()
    exp = get_experiments()

    for size in exp.keys():
        path_list = []
        for name in exp[size]:
            trajectory = get(name)

            # Variables you can access
            # trajectory.position
            # trajectory.angle
            # trajectory.fps gives frame rate

            # TODO: Plot position (x, t), plot (v, t) (-> matplotlib.pyplot), v = np.sqrt((v_x**2 + v_y**2)
            v = velocity(trajectory.position, trajectory.fps)

            # TODO: smooth_v, plot v
            smoothed_v = smooth_v(v)

            # TODO: Write function v_max()
            v_max = calculate_v_max(smoothed_v)

            # TODO: Write function find_acceleration_frames
            frames = acceleration_frames(smoothed_v, v_max)

            path_list.append(PathLength(trajectory).calculate_path_length(frames=frames))

        # resolution_dict[size] = resolution(size, shape, solver, geometry, np.mean(path_list))
        resolution_dict[size] = np.mean(path_list)
    print(resolution_dict)
