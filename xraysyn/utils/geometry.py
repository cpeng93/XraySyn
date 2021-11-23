import numpy as np
import torch

def SE3(u, v):
    G1 = np.zeros((4, 4))
    G1[0, 3] = 1

    G2 = np.zeros((4, 4))
    G2[1, 3] = 1

    G3 = np.zeros((4, 4))
    G3[2, 3] = 1

    G4 = np.zeros((4, 4))
    G4[:3, :3] = np.array(
        [
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]
    )

    G5 = np.zeros((4, 4))
    G5[:3, :3] = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0]
        ]
    )

    G6 = np.zeros((4, 4))
    G6[:3, :3] = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
    )
    return scipy.linalg.expm(
        u[0] * G1 + u[1] * G2 + u[2] * G3 + v[0] * G4 + v[1] * G5 + v[2] * G6
    )


def se3(T):
    t = scipy.linalg.logm(T)
    return (
        np.array([t[0, 3], t[1, 3], t[2, 3],]),
        np.array([t[2, 1], t[0, 2], t[1, 0]])
    )


def so3(R):
    t = scipy.linalg.logm(R)
    return np.array([t[2, 1], t[0, 2], t[1, 0]])


def get_6dofs_transformation_matrix(u, v):
    """ https://arxiv.org/pdf/1611.10336.pdf
    """
    x, y, z = u
    theta_x, theta_y, theta_z = v

    # rotate theta_z
    rotate_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_y
    rotate_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_x and translate x, y, z
    rotate_x_translate_xyz = np.array([
        [1, 0, 0, x],
        [0, np.cos(theta_x), -np.sin(theta_x), y],
        [0, np.sin(theta_x), np.cos(theta_x), z],
        [0, 0, 0, 1]
    ])

    return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)

def one_vect(batch_size, position, device):
    vect = torch.zeros(batch_size, 4)
    vect[position] = torch.ones(1)
    return vect.to(device)


# def get_6dofs_transformation_matrix_torch(v):
#     """ https://arxiv.org/pdf/1611.10336.pdf
#     """
#     device = v.device()
#     theta_x, theta_y, theta_z = v[...,0], v[...,1], v[...,2]
#     sin_x = torch.sin(theta_x)
#     cos_x = torch.cos(theta_x)
#     sin_y = torch.sin(theta_y)
#     cos_y = torch.cos(theta_y)
#     sin_z = torch.sin(theta_z)
#     cos_z = torch.cos(theta_z)



#     # rotate theta_z
#     rotate_z = np.array([
#         [torch.cos(theta_z), -np.sin(theta_z), 0, 0],
#         [np.sin(theta_z), np.cos(theta_z), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])

#     # rotate theta_y
#     rotate_y = np.array([
#         [np.cos(theta_y), 0, np.sin(theta_y), 0],
#         [0, 1, 0, 0],
#         [-np.sin(theta_y), 0, np.cos(theta_y), 0],
#         [0, 0, 0, 1]
#     ])

#     # rotate theta_x and translate x, y, z
#     rotate_x_translate_xyz = np.array([
#         [1, 0, 0, 0],
#         [0, np.cos(theta_x), -np.sin(theta_x), 0],
#         [0, np.sin(theta_x), np.cos(theta_x), 0],
#         [0, 0, 0, 1]
#     ])

#     return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)