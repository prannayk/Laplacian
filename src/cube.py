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
    def __init__(self, r=1, R1=1, R2=1, theta_base=0, phi_base=0, gamma_base=0) : # three degrees of rotation
        self.r = r
        self.R1 = R1
        self.R2 = R2
        self.rmax = np.sqrt(3)

    def _multi_point_generator(self, num):
        x = list(zip(np.random.uniform(0, self.R2, num), np.random.uniform(0, self.r, num), np.random.uniform(0, self.R1, num)))
        return np.array(x) # np.array(map(lambda x : np.matmul(self.mat1, np.array([i for i in reversed([self.r*np.cos(x[0]), (self.R1)*np.sin(x[0])*np.sin(x[1]), (self.R2*np.sin(x[0])*np.cos(x[1]))])])),list(set(x)))) # ensure there are no double matches

    def _euclidean_inv_metric(self, p1, p2, h):
        if (p1 == p2).all() : return 0
        a = np.sqrt(np.sum(np.square(p1-p2)))
        return 1.0/a

    def _euclidean_step_metric(self, p1, p2, h):
        if (p1 == p2).all() : return 0
        a = np.sqrt(np.sum(np.square(p1-p2)))
        if (a < (self.rmax*h)) :
            return 1.0/a
        else :
            return 0

    def _euclidean_metric(self, p1, p2, h):
        if (p1 == p2).all() : return 0
        return 1.0/np.sqrt(np.sum(np.square(p1-p2))/h**2)

    def _laplacian_matrix(self, points, h, lambda_val):
        kernel_pairwise = np.zeros([points.shape[0]]*2).astype(np.float64)
        zero = np.zeros(3)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if flags.metric == "euclidean_step":
                    kernel_pairwise[i,j] = self._euclidean_step_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "euclidean_inv":
                    kernel_pairwise[i,j] = self._euclidean_inv_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "euclidean":
                    kernel_pairwise[i,j] = self._euclidean_metric(points[i] , points[j], h)/h**2
                    if (kernel_pairwise[i,j] < 0):
                        print(str(points[i]) + " :: " + str(points[j]))
                    assert(kernel_pairwise[i,j] >= 0)
                else:
                    print("Invalid metric type")
                    exit(1)
        kernel_norm = np.mean(kernel_pairwise, axis=1)
        adjacency = np.array([[kernel_pairwise[i][j] / (points.shape[0]*(kernel_norm[i]*kernel_norm[j])**lambda_val) for j in range(points.shape[0])] for i in range(points.shape[0])])
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                assert(adjacency[i,j] >= 0)
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
    max_list = []
    for i in range(10):
        print("Doing for " + str(i), end='\r')
        list_pts = sphere._multi_point_generator(200)
        w,v = np.linalg.eig(sphere._laplacian_matrix(list_pts, 0.1, 0))
        max_list.append(np.sort(w)[-1])
        print("Done for " + str(i))
    print(np.mean(np.array(max_list)))
