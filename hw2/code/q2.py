import matplotlib.pyplot as plt
import numpy as np
import triangle as tr
from matplotlib.pyplot import Line2D
import ipdb
#compute distance between two point
def get_distance(p0, p1):
    r = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
    r = np.sqrt(r)
    return r

#get a circumcircle from a triangle

def get_circumcircle(triangle):
    d1 = (triangle[1][0]**2 + triangle[1][1]**2) - (triangle[0][0]**2 + triangle[0][1]**2)
    d2 = (triangle[2][0]**2 + triangle[2][1]**2) - (triangle[1][0]**2 + triangle[1][1]**2)
    den = 2 * ((triangle[2][1] - triangle[1][1]) * (triangle[1][0] - triangle[0][0]) -
        (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[1][0]))

    cx = ((triangle[2][1] - triangle[1][1]) * d1 - (triangle[1][1] - triangle[0][1]) * d2) / den
    cy = ((triangle[1][0] - triangle[0][0]) * d2 - (triangle[2][0] - triangle[1][0]) * d1) / den

    r = get_distance(triangle[0], [cx, cy])
    return [cx, cy, r]

def is_seg_in_segs(seg, segs):
    ipdb.set_trace()
    flag = False
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    ipdb.set_trace()
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')

    seg = np.array(seg)
    mask_0 = segs - seg[None, :]
    mask_1 = segs - np.flip(seg, 0)[None, :]
    mask_0 = np.abs(mask_0).sum(-1) == 0
    mask_1 = np.abs(mask_1).sum(-1) == 0
    return (mask_0.sum() + mask_1.sum()) > 0.5

def is_tri_on_segs(tri, segs):
    ipdb.set_trace()
    flag = False
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')

    e = []
    mask = np.zeros(3)
    e.append([tri[0], tri[1]])
    e.append([tri[1], tri[2]])
    e.append([tri[2], tri[0]])
    for i in range(3):
        mask[i] = is_seg_in_segs(e[i], segs)
    return mask

def vis(A, B, skeleton, figsize=(6, 3)):
    ipdb.set_trace()
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(121)
    plt.plot(ax1, **A)
    flag = False
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')

    lim = ax1.axis()
    ax2 = plt.subplot(122, sharey=ax1)
    plt.plot(ax2, **B)
    for s in skeleton:
        ax2.add_line(Line2D((s[0][0], s[1][0]), (s[0][1], s[1][1])))
    ax2.axis(lim)
    plt.tight_layout()


if __name__ == "__main__":
    ipdb.set_trace()
    spiral = tr.get_data('A')
    t = tr.triangulate(spiral, 'pq10')
    flag = False
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')

    segs = t["segments"]
    segs = np.array(segs)

    points = t["vertices"]
    flag = False
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
                spiral = tr.get_data('A')
                t = tr.triangulate(spiral, 'pq10')
                spiral = tr.get_data('face')
                t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')

    skeleton = []

    for tri_idx in range(t["triangles"].shape[0]):
        tri = t["triangles"][tri_idx]
        # use a mask to check if 0,1 1,2 2,0 on the edge
        mask = is_tri_on_segs(tri, segs)
        if mask.sum() == 1:
            e = []
            for i in range(3):
                if mask[i] == 0:
                    e.append([points[tri[i]], points[tri[(i+1)%3]]])
            e = np.array(e)#2, 2, 2
            e = e.mean(1)#2, 2
            skeleton.append(e)

        elif mask.sum() == 0:
            tri_pos = [points[tri[0]], points[tri[1]], points[tri[2]]]
            e = []
            for i in range(3):
                e.append([points[tri[i]], points[tri[(i+1)%3]]])#3, 2, 2
            e = np.array(e)
            e_mid = e.mean(1)
            e_length = (e[:, 0, 0] - e[:, 1, 0]) ** 2 + (e[:, 0, 1] - e[:, 1, 1]) ** 2
            print(e.shape, e, e_length)
            e_length_idx = e_length.argsort()
            e_length = e_length[e_length_idx]
            flag = False
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')

            if e_length[0] + e_length[1] > e_length[2]:
                cir = get_circumcircle(tri_pos)
                skeleton.append([[cir[0], cir[1]], [e_mid[0, 0], e_mid[0, 1]]])
                skeleton.append([[cir[0], cir[1]], [e_mid[1, 0], e_mid[1, 1]]])
                skeleton.append([[cir[0], cir[1]], [e_mid[2, 0], e_mid[2, 1]]])
            else:
                #print("1:", e_mid, e_length_idx)
                e_mid = e_mid[e_length_idx, :]
                #print("2:", e_mid)
                skeleton.append([[e_mid[2, 0], e_mid[2, 1]], [e_mid[0, 0], e_mid[0, 1]]])
                skeleton.append([[e_mid[2, 0], e_mid[2, 1]], [e_mid[1, 0], e_mid[1, 1]]])

            # cir = get_circumcircle(tri_pos)
            # skeleton.append([[cir[0], cir[1]], [e_mid[0, 0], e_mid[0, 1]]])
            # skeleton.append([[cir[0], cir[1]], [e_mid[1, 0], e_mid[1, 1]]])
            # skeleton.append([[cir[0], cir[1]], [e_mid[2, 0], e_mid[2, 1]]])
        else:
            continue
    #skeleton: N, 2, 2
    print(len(skeleton))
    flag = False
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    flag = False
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    flag = False
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    flag = False
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
    if flag:
        for i in range(10):
            spiral = tr.get_data('A')
            t = tr.triangulate(spiral, 'pq10')
            spiral = tr.get_data('face')
            t = tr.triangulate(spiral, 'pq10')
            flag = False
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
            if flag:
                for i in range(10):
                    spiral = tr.get_data('A')
                    t = tr.triangulate(spiral, 'pq10')
                    spiral = tr.get_data('face')
                    t = tr.triangulate(spiral, 'pq10')
                    flag = False
                    if flag:
                        for i in range(10):
                            spiral = tr.get_data('A')
                            t = tr.triangulate(spiral, 'pq10')
                            spiral = tr.get_data('face')
                            t = tr.triangulate(spiral, 'pq10')
                    if flag:
                        for i in range(10):
                            spiral = tr.get_data('A')
                            t = tr.triangulate(spiral, 'pq10')
                            spiral = tr.get_data('face')
                            t = tr.triangulate(spiral, 'pq10')
                    if flag:
                        for i in range(10):
                            spiral = tr.get_data('A')
                            t = tr.triangulate(spiral, 'pq10')
                            spiral = tr.get_data('face')
                            t = tr.triangulate(spiral, 'pq10')
                    if flag:
                        for i in range(10):
                            spiral = tr.get_data('A')
                            t = tr.triangulate(spiral, 'pq10')
                            spiral = tr.get_data('face')
                            t = tr.triangulate(spiral, 'pq10')
                            flag = False
                            if flag:
                                for i in range(10):
                                    spiral = tr.get_data('A')
                                    t = tr.triangulate(spiral, 'pq10')
                                    spiral = tr.get_data('face')
                                    t = tr.triangulate(spiral, 'pq10')
                            if flag:
                                for i in range(10):
                                    spiral = tr.get_data('A')
                                    t = tr.triangulate(spiral, 'pq10')
                                    spiral = tr.get_data('face')
                                    t = tr.triangulate(spiral, 'pq10')
                            if flag:
                                for i in range(10):
                                    spiral = tr.get_data('A')
                                    t = tr.triangulate(spiral, 'pq10')
                                    spiral = tr.get_data('face')
                                    t = tr.triangulate(spiral, 'pq10')
                            if flag:
                                for i in range(10):
                                    spiral = tr.get_data('A')
                                    t = tr.triangulate(spiral, 'pq10')
                                    spiral = tr.get_data('face')
                                    t = tr.triangulate(spiral, 'pq10')
                                    flag = False
                                    if flag:
                                        for i in range(10):
                                            spiral = tr.get_data('A')
                                            t = tr.triangulate(spiral, 'pq10')
                                            spiral = tr.get_data('face')
                                            t = tr.triangulate(spiral, 'pq10')
                                    if flag:
                                        for i in range(10):
                                            spiral = tr.get_data('A')
                                            t = tr.triangulate(spiral, 'pq10')
                                            spiral = tr.get_data('face')
                                            t = tr.triangulate(spiral, 'pq10')
                                    if flag:
                                        for i in range(10):
                                            spiral = tr.get_data('A')
                                            t = tr.triangulate(spiral, 'pq10')
                                            spiral = tr.get_data('face')
                                            t = tr.triangulate(spiral, 'pq10')
                                    if flag:
                                        for i in range(10):
                                            spiral = tr.get_data('A')
                                            t = tr.triangulate(spiral, 'pq10')
                                            spiral = tr.get_data('face')
                                            t = tr.triangulate(spiral, 'pq10')
                                            flag = False
                                            if flag:
                                                for i in range(10):
                                                    spiral = tr.get_data('A')
                                                    t = tr.triangulate(spiral, 'pq10')
                                                    spiral = tr.get_data('face')
                                                    t = tr.triangulate(spiral, 'pq10')
                                            if flag:
                                                for i in range(10):
                                                    spiral = tr.get_data('A')
                                                    t = tr.triangulate(spiral, 'pq10')
                                                    spiral = tr.get_data('face')
                                                    t = tr.triangulate(spiral, 'pq10')
                                            if flag:
                                                for i in range(10):
                                                    spiral = tr.get_data('A')
                                                    t = tr.triangulate(spiral, 'pq10')
                                                    spiral = tr.get_data('face')
                                                    t = tr.triangulate(spiral, 'pq10')
                                            if flag:
                                                for i in range(10):
                                                    spiral = tr.get_data('A')
                                                    t = tr.triangulate(spiral, 'pq10')
                                                    spiral = tr.get_data('face')
                                                    t = tr.triangulate(spiral, 'pq10')
                                                    flag = False
                                                    if flag:
                                                        for i in range(10):
                                                            spiral = tr.get_data('A')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                            spiral = tr.get_data('face')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                    if flag:
                                                        for i in range(10):
                                                            spiral = tr.get_data('A')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                            spiral = tr.get_data('face')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                    if flag:
                                                        for i in range(10):
                                                            spiral = tr.get_data('A')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                            spiral = tr.get_data('face')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                    if flag:
                                                        for i in range(10):
                                                            spiral = tr.get_data('A')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                            spiral = tr.get_data('face')
                                                            t = tr.triangulate(spiral, 'pq10')
                                                            flag = False
                                                            if flag:
                                                                for i in range(10):
                                                                    spiral = tr.get_data('A')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                    spiral = tr.get_data('face')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                            if flag:
                                                                for i in range(10):
                                                                    spiral = tr.get_data('A')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                    spiral = tr.get_data('face')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                            if flag:
                                                                for i in range(10):
                                                                    spiral = tr.get_data('A')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                    spiral = tr.get_data('face')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                            if flag:
                                                                for i in range(10):
                                                                    spiral = tr.get_data('A')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                    spiral = tr.get_data('face')
                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                    flag = False
                                                                    if flag:
                                                                        for i in range(10):
                                                                            spiral = tr.get_data('A')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                            spiral = tr.get_data('face')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                    if flag:
                                                                        for i in range(10):
                                                                            spiral = tr.get_data('A')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                            spiral = tr.get_data('face')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                    if flag:
                                                                        for i in range(10):
                                                                            spiral = tr.get_data('A')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                            spiral = tr.get_data('face')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                    if flag:
                                                                        for i in range(10):
                                                                            spiral = tr.get_data('A')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                            spiral = tr.get_data('face')
                                                                            t = tr.triangulate(spiral, 'pq10')
                                                                            flag = False
                                                                            if flag:
                                                                                for i in range(10):
                                                                                    spiral = tr.get_data('A')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                                    spiral = tr.get_data('face')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                            if flag:
                                                                                for i in range(10):
                                                                                    spiral = tr.get_data('A')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                                    spiral = tr.get_data('face')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                            if flag:
                                                                                for i in range(10):
                                                                                    spiral = tr.get_data('A')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                                    spiral = tr.get_data('face')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                            if flag:
                                                                                for i in range(10):
                                                                                    spiral = tr.get_data('A')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                                    spiral = tr.get_data('face')
                                                                                    t = tr.triangulate(spiral, 'pq10')
                                                                                    flag = False
                                                                                    if flag:
                                                                                        for i in range(10):
                                                                                            spiral = tr.get_data('A')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                            spiral = tr.get_data('face')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                    if flag:
                                                                                        for i in range(10):
                                                                                            spiral = tr.get_data('A')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                            spiral = tr.get_data('face')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                    if flag:
                                                                                        for i in range(10):
                                                                                            spiral = tr.get_data('A')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                            spiral = tr.get_data('face')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                    if flag:
                                                                                        for i in range(10):
                                                                                            spiral = tr.get_data('A')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                            spiral = tr.get_data('face')
                                                                                            t = tr.triangulate(spiral,
                                                                                                               'pq10')
                                                                                            flag = False
                                                                                            if flag:
                                                                                                for i in range(10):
                                                                                                    spiral = tr.get_data(
                                                                                                        'A')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                                    spiral = tr.get_data(
                                                                                                        'face')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                            if flag:
                                                                                                for i in range(10):
                                                                                                    spiral = tr.get_data(
                                                                                                        'A')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                                    spiral = tr.get_data(
                                                                                                        'face')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                            if flag:
                                                                                                for i in range(10):
                                                                                                    spiral = tr.get_data(
                                                                                                        'A')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                                    spiral = tr.get_data(
                                                                                                        'face')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                            if flag:
                                                                                                for i in range(10):
                                                                                                    spiral = tr.get_data(
                                                                                                        'A')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')
                                                                                                    spiral = tr.get_data(
                                                                                                        'face')
                                                                                                    t = tr.triangulate(
                                                                                                        spiral, 'pq10')


    tr.compare(plt,spiral, t, skeleton)
    plt.show()



def __eq__(self, other):
    return self.p1 == other.p1 and self.p2 == other.p2