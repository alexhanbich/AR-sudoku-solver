import numpy as np

# order: top_left, top_right, bottom_left, bottom_right
def sort_points(coords):
    sorted_by_y = sorted(coords, key=lambda x:x[1])
    sorted_top_by_x = sorted(sorted_by_y[2:], key=lambda x:x[0])
    sorted_bottom_by_x = sorted(sorted_by_y[:2], key=lambda x:x[0])
    sorted_points = np.concatenate((sorted_top_by_x, sorted_bottom_by_x))
    return sorted_points
    


coor1 = np.array([(0,2), (2,3), (1,0), (3,1)])
print(sort_points(coor1))

coor2 = np.array([(2,0), (1,3), (3,2), (0,1)])
print(sort_points(coor2))