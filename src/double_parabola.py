'''
The Sphere manifold can be paramterized using theta and phi
The sphere manifold is isometric to R2 using the conical projection map
The Sphere manifold's laplacian hence must be similar to the laplacian of R2
This experiment would seek to infer the laplacian from given samples, with the manifold being endowed with an inherent metric
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

class Sphere(object):
    def __init__(self, ax=1, bx=0, cx=0, ay=0, by=1, cy=0, az=0, bz=0, cz=1, tmax=10) : # three degrees of rotation
        self.matmul1 = np.zeros([3,3])
        self.matmul2 = np.zeros([3,3])
        self.matmula[0,0] = ax
        self.matmula[0,1] = bx
        self.matmula[0,2] = cx
        self.matmula[1,0] = ay
        self.matmula[1,1] = by
        self.matmula[1,2] = cy
        self.matmula[2,0] = az
        self.matmula[2,1] = bz
        self.rmax=1.0
        self.tmax = tmax

    def _multi_point_generator(self, num):
        x = np.random.uniform(0, self.tmax, num)
        return np.array(map(lambda x : np.matmul(self.matmul, np.array([x**2, x, 1])),list(set(x)))) # ensure there are no double matches

    def _euclidean_step_metric(self, p1, p2, h):
        if (p1 == p2).all() : return 0
        a = np.sqrt(np.sum(np.square(p1-p2)))
        if a < self.rmax*h :
            return 1.0/a
        else :
            return 0

    def _parabolic_inv_metric(self, p1, p2, h):
        if np.array_equal(p1,p2) : return 0
        a = self._parabolic_metric(p1, p2)
        return 1.0/a
    def _euclidean_inv_metric(self, p1, p2, h):
        if np.array_equal(p1,p2) : return 0
        a = np.sqrt(np.sum(np.square(p1-p2)))
        return 1.0/a

    def _parabolic_step_metric(self, p1, p2, h):
        if np.array_equal(p1,p2) : return 0
        a = self._parabolic_metric(p1, p2)
        if a < self.rmax*h :
            return 1.0/a
        else :
            return 0

    def _parabolic_metric(self, p1, p2):
        p1 = np.matmul(np.linalg.inv(self.matmul), p1)
        p2 = np.matmul(np.linalg.inv(self.matmul), p2)
        t1 = p1[0] / p1[1]
        t2 = p2[0] / p2[1]
        return np.abs(t1 -t2)/10

    def _euclidean_metric(self, p1, p2, h):
        if (p1 == p2).all() : return 0
        return 1/np.sqrt(np.sum(np.square(p1-p2))/h**2)

    def _laplacian_matrix(self, points, h, lambda_val):
        kernel_pairwise = np.zeros([points.shape[0]]*2).astype(np.float64)
        zero = np.zeros(3)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if flags.metric == "sphere":
                    exit(1)
                    kernel_pairwise[i,j] = self._sphere_metric(points[i] , points[j], h)/h
                elif flags.metric == "cosine":
                    exit(1)
                    kernel_pairwise[i,j] = self._cosine_metric(points[i] , points[j], h)/h
                elif flags.metric == "parabolic_inv":
                    kernel_pairwise[i,j] = self._parabolic_inv_metric(points[i] , points[j], h)/h
                elif flags.metric == "parabolic_step":
                    kernel_pairwise[i,j] = self._parabolic_step_metric(points[i] , points[j], h)/h
                elif flags.metric == "euclidean_inv":
                    kernel_pairwise[i,j] = self._euclidean_inv_metric(points[i] , points[j], h)/h
                elif flags.metric == "euclidean_step":
                    kernel_pairwise[i,j] = self._euclidean_step_metric(points[i] , points[j], h)/h
                elif flags.metric == "euclidean":
                    kernel_pairwise[i,j] = self._euclidean_metric(points[i] , points[j], h)/h
                    if (kernel_pairwise[i,j] < 0):
                        print(str(points[i]) + " :: " + str(points[j]))
                    assert(kernel_pairwise[i,j] >= 0)
                else:
                    print("Invalid metric type")
                    exit(1)
        kernel_norm = np.mean(kernel_pairwise, axis=1)
        adjacency = np.array([[kernel_pairwise[i][j] / (points.shape[0]*(kernel_norm[i]*kernel_norm[j])**lambda_val) for j in range(points.shape[0])] for i in range(points.shape[0])])
        diagonal = np.diag(np.sum(adjacency, axis=1))
        diag_sqrt = np.sqrt(diagonal)
        diag_sqrt_inv = np.linalg.inv(diag_sqrt)
        norm_adj = np.matmul(diag_sqrt_inv, np.matmul(adjacency, diag_sqrt_inv))
        laplacian = (1/h**2)*np.matmul(diag_sqrt_inv, np.matmul(diagonal - adjacency, diag_sqrt_inv))
        return laplacian


tf.app.flags.DEFINE_string("metric", "sphere", "Metric for computation")
flags = tf.app.flags.FLAGS

if __name__ == "__main__" :
    sphere = Sphere()
    zero = np.zeros(3)
    list_pts = sphere._multi_point_generator(50)
    for num in [50,100,150,200,250]:
        list_pts = sphere._multi_point_generator(num)
        for h in [0.1]: # 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
            for lambda_val in [0,1,2,3,4,5,6]:
                w,v = np.linalg.eig(sphere._laplacian_matrix(list_pts, h, lambda_val))
                w = w / (2*2)
                print("Printing h, lambda, largest eigen : " + str(h) + " , " + str(lambda_val)  + " , " + str(np.sort(w)[-1]))
