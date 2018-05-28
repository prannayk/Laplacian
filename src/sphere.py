'''
The Sphere manifold can be paramterized using theta and phi
The sphere manifold is isometric to R2 using the conical projection map
The Sphere manifold's laplacian hence must be similar to the laplacian of R2
This experiment would seek to infer the laplacian from given samples, with the manifold being endowed with an inherent metric
'''

from __future__ import print_function
import numpy as np

class Sphere(object):
    def __init__(self):
        self.theta_range = np.pi
        self.phi_range = 2*np.pi

    def _single_point_generator(self):
        x = np.random.uniform(0, self.theta_range), np.random.uniform(0, self.phi_range)
        return reversed([np.cos(x[0]), (np.sin(x[0])*np.sin(x[1])), (np.sin(x[0])*np.cos(x[1]))])

    def _multi_point_generator(self, num):
        x = zip(np.random.uniform(0, self.theta_range, num), np.random.uniform(0, self.phi_range, num))
        return np.array(map(lambda x : [i for i in reversed([np.cos(x[0]), (np.sin(x[0])*np.sin(x[1])), (np.sin(x[0])*np.cos(x[1]))])],list(set(x)))) # ensure there are no double matches

    def _euclidean_metric(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1-p2)))

    def _sphere_metric(self, p1, p2, h):
        theta1 = np.arctan(np.sqrt(p1[0]**2 + p1[1]**2) / p1[2])
        phi1 = np.arctan(p1[1]/p1[0])
        theta2 = np.arctan(np.sqrt(p2[0]**2 + p2[1]**2) / p2[2])
        phi2 = np.arctan(p2[1]/p2[0])
        val = np.square(theta1 - theta2) / (np.pi*np.pi)
        val += np.square(phi1 - phi2) / (4*np.pi*np.pi)
        return np.sqrt(val/ h**2)


    def _laplacian_matrix(self, points, h, lambda_val):
        kernel_pairwise = np.zeros([points.shape[0]]*2)
        zero = np.zeros(3)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                kernel_pairwise[i,j] = self._sphere_metric(points[i] , points[j], h)/h**2
        kernel_norm = np.mean(kernel_pairwise, axis=1)
        adjacency = np.array([[kernel_pairwise[i][j] / (points.shape[0]*(kernel_norm[i]*kernel_norm[j])**lambda_val) for j in range(points.shape[0])] for i in range(points.shape[0])])
        d_matrix = np.sum(adjacency, axis=1)*np.ones([points.shape[0],1])
        diagonal = np.diag(np.sum(adjacency, axis=1))
        premup = np.ones([points.shape[0], points.shape[0]]) / d_matrix
        laplacian = (1/h**2)*(diagonal -  adjacency)
        return laplacian




sphere = Sphere()
zero = np.zeros(3)
list_pts = sphere._multi_point_generator(200)
w,v = np.linalg.eig(sphere._laplacian_matrix(list_pts, 0.2,1))
w = w / (2*2)
print(np.sort(w)[:2])
print(np.sort(w)[-1:])