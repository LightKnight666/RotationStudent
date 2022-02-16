from trajectory_inheritance.trajectory import get, Trajectory
from Analysis.PathLength import PathLength
from matplotlib import pyplot as plt
from Directories import directory
import os


def typical_v_max() -> float:
    # TODO Rotation students: Find typical top speed
    return float()
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

            # trajectory.position
            # trajectory.angle
            # trajectory.angle.__class__ gives you the class of your object

            # TODO Itay: Plot position (x, t), plot (speed, t) (-> matplotlib.pyplot)
            # plt.figure()
            # plt.plot(trajectory.angle)

            # frames = find_acceleration_frames(traj)
            frames = [1, 100]
            path_list.append(PathLength(trajectory).calculate_path_length(frames=frames))

        # resolution_dict[size] = resolution(size, shape, solver, geometry, np.mean(path_list))
    print(resolution_dict)

