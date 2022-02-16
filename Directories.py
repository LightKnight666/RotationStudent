from os import path

home = path.join(path.abspath(__file__).split('\\')[0]+path.sep, *path.abspath(__file__).split(path.sep)[1:-1])

directory = path.join(home, 'experimental_trajectories', 'Human_Trajectories')
