from trajectory_inheritance.trajectory import Trajectory_part
import numpy as np
from copy import copy
from trajectory_inheritance.trajectory import get
from matplotlib import pyplot as plt


# --- from experimental data--- #
# StartedScripts: check the noises (humans!!!)
noise_xy_ants_ImageAnaylsis = [0.01, 0.05, 0.02]  # cm
noise_angle_ants_ImageAnaylsis = [0.01, 0.01, 0.02]  # rad

noise_xy_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # m
noise_angle_human_ImageAnaylsis = [0.01, 0.01, 0.01]  # rad

resolution_xy_of_ps = [0.08, 0.045]  # cm
resolution_angle_of_ps = [0.05, 0.05]  # cm
periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}


def resolution(geometry: tuple, size: str, solver: str, shape: str):
    if geometry == ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'):
        Res = {'Large': 0.1, 'L': 0.1, 'Medium': 0.07, 'M': 0.07, 'Small Far': 0.02, 'Small Near': 0.02, 'S': 0.02}
        return Res[size] * 0.5  # had to add twice the resolution, because the fast marching method was not able to pass
    else:
        raise ValueError


def ConnectAngle(angle, shape):
    # This function writes the angle as absolute angle compared to the beginning angle (which was within a certain 0
    # to 2pi range) Write the first angle in the connected angle array

    ''' turn the shape by 2/p * np.pi and it looks the same'''
    p = periodicity[shape]

    ''' Wrap it! '''
    angle = (angle + np.pi / p) % (2 * np.pi / p) - np.pi / p

    ''' Make NaNs and the adjacent 5 values NaNs'''
    original_nan = np.where(np.isnan(angle))[0]

    for i in range(5):
        if len(original_nan) > 0 and original_nan[-1] - i > 0:
            angle[original_nan - i] = np.NaN
        elif len(original_nan) > 0 and len(angle) - 1 > original_nan[-1] + i:
            angle[original_nan + i] = np.NaN
    not_new_nan = ~np.isnan(angle)

    ''' get rid of all NaNs '''
    angle = angle[~np.isnan(angle)]

    # ok = ~np.isnan(angle)
    # xp = ok.ravel().nonzero()[0]
    # fp = angle[~np.isnan(angle)]
    # x  = np.isnan(angle).ravel().nonzero()[0]

    ''' unwrap '''
    # angle[np.isnan(angle)] = np.interp(x, xp, fp)
    unwraped = 1 / p * np.unwrap(p * angle)
    returner = np.empty([len(not_new_nan)])

    ''' reinsert NaNs '''
    i, ii = 0, 0
    for insert in not_new_nan:
        if insert:
            returner[ii] = unwraped[i]
            i = i + 1
        else:
            returner[ii] = np.NaN
        ii = ii + 1
    # unwraped[originalNaN[0]] = np.NaN

    return returner


class PathLength:
    def __init__(self, x):
        self.x = copy(x)

    def during_attempts(self, *args, attempts=None, **kwargs):
        return None
    #     # TODO
    #     """
    #     Path length is calculated during attempts.
    #     End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
    #     """
    #     return None
    #     total = 0
    #
    #     if attempts is None:
    #         attempts = Attempts(self.x, 'extend', *args, **kwargs)
    #
    #     for attempt in attempts:
    #         total += self.calculate_path_length(start=attempt[0], end=attempt[1])
    #     # if total < Maze(size=x.size, shape=x.shape, solver=x.solver).minimal_path_length:
    #     #     print(x)
    #     #     print(total)
    #     #     print(x.filename + ' was smaller than the minimal path length... ')
    #     return total

    def per_experiment(self) -> float:
        """
        Path length is calculated from beginning to end_screen.
        End is either given through the kwarg 'minutes', or is defined as the end_screen of the experiment.
        """
        # I have to split movies, because for 'connector movies', we have to treat them separately.
        parts = self.x.divide_into_parts()
        path_lengths = [PathLength(part).calculate_path_length() for part in parts]
        interpolated_path_lengths = self.interpolate_connectors(parts, path_lengths)
        return np.sum(interpolated_path_lengths)

    def interpolate_connectors(self, parts, path_lengths) -> list:
        """
        :param parts: parts of trajectories
        :param path_lengths: calculated path lengths which contain nans.
        :return: path lengths without nans.
        """
        missed_frames = np.sum([len(part.frames) for part, path_length in
                                zip(parts, path_lengths) if np.isnan(path_length)])
        total_length = self.x.frames.shape[0]
        path_length_per_frame = np.nansum(path_lengths)/(total_length - missed_frames)
        return [len(part.frames) * path_length_per_frame
                if np.isnan(path_length) else path_length for part, path_length in zip(parts, path_lengths)]

    @staticmethod
    def measureDistance(position1, position2, angle1, angle2, averRad, rot=True, **kwargs):  # re`turns distance in cm.
        archlength = 0
        if position1.ndim == 1:  # For comparing only two positions
            translation = np.linalg.norm(position1[:2] - position2[:2])
            if rot:
                archlength = abs(angle1 - angle2) * averRad

        else:  # For comparing more than 2 positions
            # translation = np.sqrt(np.sum(np.power((position1[:, :2] - position2[:, :2]), 2), axis=1))
            translation = np.linalg.norm(position1[:, :2] - position2[:, :2])
            if rot:
                archlength = abs(angle1[:] - angle2[:]) * averRad
        return translation + archlength

    def average_radius(self):
        av_radius = {'Large': 4.899, 'Medium': 2.441, 'Small Far': 1.198, 'Small Near': 1.198}  # this is only for
        # humans
        return av_radius[self.x.size]

    def calculate_path_length(self, rot: bool = True, frames: list = None):
        """
        Reduce path to a list of points that each have distance of at least resolution = 0.1cm
        to the next point.
        Distance between to points is calculated by |x1-x2| + (angle1-angle2) * aver_radius.
        Path length the sum of the distances of the points in the list.
        """
        if frames is None:
            frames = [0, -1]

        # the connector parts have to short of a path length.
        if isinstance(self.x, Trajectory_part) and self.x.is_connector():
            return np.NaN

        position, angle = self.x.position[frames[0]: frames[1]], self.x.angle[frames[0]: frames[1]]
        aver_radius = self.average_radius()

        unwrapped_angle = ConnectAngle(angle[1:], self.x.shape)
        if unwrapped_angle.size == 0 or position.size == 0:
            return 0
        pos, ang = position[0], unwrapped_angle[0]
        path_length = 0

        for i in range(len(unwrapped_angle)):
            d = self.measureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
            if d > resolution(self.x.geometry(), self.x.size, self.x.solver, self.x.shape):
                path_length += self.measureDistance(pos, position[i], ang, unwrapped_angle[i], aver_radius, rot=rot)
                pos, ang = position[i], unwrapped_angle[i]
        return path_length

    def plot(self, rot=True):
        plt.plot(self.x.position[:, 0], self.x.position[:, 1], color='blue')
        pos_list, ang_list = [], []

        unwrapped_angle = ConnectAngle(self.x.angle[1:], self.x.shape)
        pos, ang = self.x.position[0], unwrapped_angle[0]
        for i in range(len(unwrapped_angle)):
            d = self.measureDistance(pos, self.x.position[i], ang, unwrapped_angle[i], self.average_radius(), rot=rot)
            if d > resolution(self.x.geometry(), self.x.size, self.x.solver, self.x.shape):
                pos_list.append(pos)
                ang_list.append(ang)
        plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1], color='k')
        plt.show()


if __name__ == '__main__':
    x = get('XL_SPT_4630002_XLSpecialT_1_ants (part 1)')
    print(PathLength(x).per_experiment())
    # p = [resolution(size, 'ant') for size in sizes['ant']]
