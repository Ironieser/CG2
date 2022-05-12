import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np

#生成输入
def get_input_ramdom(input_num):
    points_x = np.random.rand(input_num)
    points_x = np.sort(points_x, axis=0)
    points_y = np.random.rand(input_num)
    points = np.stack([points_x, points_y], axis=1)
    return points

#计算两点距离
def get_distance(p0, p1):
    r = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
    r = np.sqrt(r)
    return r

#得到一个三角形的外接圆
def get_circumcircle(triangle):
    d1 = (triangle[1][0]**2 + triangle[1][1]**2) - (triangle[0][0]**2 + triangle[0][1]**2)
    d2 = (triangle[2][0]**2 + triangle[2][1]**2) - (triangle[1][0]**2 + triangle[1][1]**2)
    den = 2 * ((triangle[2][1] - triangle[1][1]) * (triangle[1][0] - triangle[0][0]) - \
        (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[1][0]))

    cx = ((triangle[2][1] - triangle[1][1]) * d1 - (triangle[1][1] - triangle[0][1]) * d2) / den
    cy = ((triangle[1][0] - triangle[0][0]) * d2 - (triangle[2][0] - triangle[1][0]) * d1) / den

    r = get_distance(triangle[0], [cx, cy])
    return [cx, cy, r]

def remove_rep(tri_or_edge):
    i = 0
    while i < len(tri_or_edge):
        for j in range(i+1, len(tri_or_edge)):
            tri_i = np.array(triangles[i])#3, 2
            tri_j = np.array(triangles[j])
            same_point_mask = np.zeros(3)
            for p in range(3):
                same_mask = tri_j - tri_i[p:p+1, :]
                same_mask = same_mask.sum(-1)
                same_mask = same_mask == 0
                if same_mask.sum() > 0:
                    same_point_mask[p] = 1
            if same_point_mask.sum == 2 or same_point_mask.sum == 3:
                triangles.pop(j)
                print("remove")
        i += 1


    return triangles

def get_delaunay_triangulation(points):
    #find a cover triangle
    triangles = []
    temp_triangles = []
    #初始化一个最大的超三角形
    super_triangles = [[-10, -1], [0.5, 10], [10, -1]]
    temp_triangles.append(super_triangles)
    #开始向图里逐步添加点
    for point_idx in range(points.shape[0]):
        edge = []
        num_tri = len(temp_triangles)
        for temp_tri_idx in range(num_tri):
            now_triangle = temp_triangles.pop(0)
            now_circle = get_circumcircle(now_triangle)
            dis_to_ctr = get_distance(points[point_idx], now_circle[:2])
            #如果点在外接圆内，把当前三角形拆成边，之后分别与点相连接
            if dis_to_ctr < now_circle[2]:
                edge.append([now_triangle[0], now_triangle[1]])
                edge.append([now_triangle[1], now_triangle[2]])
                edge.append([now_triangle[2], now_triangle[0]])
            #如果点在外接圆右侧
            elif points[point_idx][0] - now_circle[0] > now_circle[2]:
                triangles.append(now_triangle)
            #点在外接圆外
            else:
                temp_triangles.append(now_triangle)

        #给边降重
        edge = np.array(edge)#N, 2, 2
        sort_idx = np.argsort(edge[:, :, 0], axis=1)[:, :, None].repeat(2, axis=2)
        f_indices = np.arange(edge.shape[0])[:, None, None]
        s_indices = np.arange(edge.shape[2])[None, None, :].repeat(edge.shape[0], axis=0).repeat(edge.shape[1], axis=1)
        edge = edge[f_indices, sort_idx, s_indices]
        edge, count = np.unique(edge, axis=0, return_counts=True)
        edge = edge[count <= 1]
        #连接当前点与所有边
        for edge_idx in range(edge.shape[0]):
            temp_triangles.append([points[point_idx], edge[edge_idx, 0], edge[edge_idx, 1]])

    #最终所有三角形，降重
    triangles += temp_triangles
    triangles = np.array(triangles)
    sort_idx = np.argsort(triangles[:, :, 0], axis=1)[:, :, None].repeat(2, axis=2)
    f_indices = np.arange(triangles.shape[0])[:, None, None]
    s_indices = np.arange(triangles.shape[2])[None, None, :].repeat(triangles.shape[0], axis=0).repeat(triangles.shape[1], axis=1)
    triangles = triangles[f_indices, sort_idx, s_indices]
    triangles, count = np.unique(triangles, axis=0, return_counts=True)#N, 3, 2
    triangles = triangles[count <= 1]
    #把超三角形相关的全删除
    for point in super_triangles:
        point = np.array(point)
        mask = triangles - point[None, None, :]
        mask = mask.sum(-1)
        mask = np.logical_and(np.logical_and(mask[:, 0] != 0., mask[:, 1] != 0.), mask[:, 2] != 0.)
        triangles = triangles[mask, ...]


    
    return triangles

#可视化
def triangles_visualization(triangles):
    fig, ax = plt.subplots(1,1)
    for t in triangles:
        # cir = get_circumcircle(t)
        # cir = Circle((cir[0], cir[1]), cir[2])
        # cir = ax.add_patch(cir)
        # cir.set_fill(False)
        t = Polygon(t)
        t = ax.add_patch(t)
        t.set_fill(False)
        
    plt.show()


if __name__ == "__main__":
    points = get_input_ramdom(50)
    triangles = get_delaunay_triangulation(points)
    print(triangles.shape[0])
    triangles_visualization(triangles)
            


