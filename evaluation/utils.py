import numpy as np
import torch

from plyfile import PlyData, PlyElement

def load_ply(filename):
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
    vertex = plydata['vertex'].data
    has_color = all(c in vertex.dtype.names for c in ['red', 'green', 'blue'])
    coords = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    normal_coords = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1)
    alpha = np.stack([vertex['alpha']], axis=-1)
    
    if has_color:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1).astype(np.float32)# / 255.0
        points = np.concatenate([coords, colors], axis=-1)  # [N, 6]
    else:
        points = coords  # [N, 3]
    return points, normal_coords, alpha

def save_ply(path, points):
    N = points.shape[0]
    #rgb_uint8 = (rgb * 255).astype(np.uint8)
    rgb_uint8 = (points[:, 6:10]).astype(np.uint8)
    mask_uint8 = (points[:, 10]).astype(np.uint8)
    verts = np.zeros(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
        ('mask', 'u1')
    ])
    verts['x'] = points[:, 0]
    verts['y'] = points[:, 1]
    verts['z'] = points[:, 2]
    verts['nx'] = points[:, 3]
    verts['ny'] = points[:, 4]
    verts['nz'] = points[:, 5]
    verts['red'] = rgb_uint8[:, 0]
    verts['green'] = rgb_uint8[:, 1]
    verts['blue'] = rgb_uint8[:, 2]
    verts['alpha'] = rgb_uint8[:, 3]
    verts['mask'] = mask_uint8[:]
    ply = PlyData([PlyElement.describe(verts, 'vertex')], text=True)
    ply.write(path)

# def load_ply(filename):
#     with open(filename, "rb") as rf:
#         while True:
#             try:
#                 line = rf.readline()
#             except:
#                 raise NotImplementedError
#             if "end_header" in line:
#                 break
#             if "element vertex" in line:
#                 arr = line.split()
#                 num_of_points = int(arr[2])

#         # print("%d points in ply file" %num_of_points)
#         points = np.zeros([num_of_points, 6])
#         for i in range(points.shape[0]):
#             point = rf.readline().split()
#             assert len(point) == 6
#             points[i][0] = float(point[0])
#             points[i][1] = float(point[1])
#             points[i][2] = float(point[2])
#             points[i][3] = float(point[3])
#             points[i][4] = float(point[4])
#             points[i][5] = float(point[5])
#         return points


# def save_ply(mesh_path, points, rgb):
#     """
#     Save the visualization of sampling to a ply file.
#     Red points represent positive predictions.
#     Green points represent negative predictions.
#     :param mesh_path: File name to save
#     :param points: [N, 3] array of points
#     :param rgb: [N, 3] array of rgb values in the range [0~1]
#     :return:
#     """
#     to_save = np.concatenate([points, rgb * 255], axis=-1)
#     return np.savetxt(
#         mesh_path,
#         to_save,
#         fmt="%.6f %.6f %.6f %d %d %d",
#         comments="",
#         header=(
#             "ply\nformat ascii 1.0\nelement vertex {:d}\n"
#             + "property float x\nproperty float y\nproperty float z\n"
#             + "property uchar red\nproperty uchar green\nproperty uchar blue\n"
#             + "end_header"
#         ).format(points.shape[0]),
#     )


def visualize_prompts(path, points, prompt, labels, atol=0.005, points_num=1000):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(prompt, torch.Tensor):
        prompt = prompt.detach().cpu().numpy()

    # sample points around prompt
    for p in prompt:
        sampled_points = []
        for _ in range(points_num):
            diff = np.random.uniform(-atol, atol, [3])
            sampled_points.append(p + diff)
        sampled_points = np.stack(sampled_points)
        points = np.concatenate([points, sampled_points], axis=0)
    colors = np.ones_like(points)
    for i in range(prompt.shape[0]):
        start = -points_num * (len(prompt) - i)
        end = (
            -points_num * (len(prompt) - i - 1)
            if -points_num * (len(prompt) - i - 1) < 0
            else -1
        )
        colors[start:end] = [1, 0, 0] if labels[i] else [0, 1, 0]
    save_ply(path, points, colors)


def visualize_mask(path, points, mask):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    colors = np.ones_like(points)
    colors[mask > 0] = [1, 0, 0]
    save_ply(path, points, colors)


def visualize_pc(path, points, rgb=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if rgb is None:
        colors = np.ones_like(points)
    else:
        colors = rgb
    save_ply(path, points, colors)
