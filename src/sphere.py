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
        self.theta_range = np.pi
        self.phi_range = 2*np.pi
        self.r = r
        self.R1 = R1
        self.R2 = R2
        self.z_mat = np.array([[np.cos(theta_base), -np.sin(theta_base),0],[np.sin(theta_base), np.cos(theta_base),0],[0 , 0, 1]]).astype(np.float32)
        self.x_mat = np.array([[1,0,0], [0, np.cos(phi_base), -np.sin(phi_base)],[0, np.sin(phi_base), np.cos(phi_base)]])
        self.y_mat = np.array([[np.cos(gamma_base),0,np.sin(gamma_base)], [0, 1,0], [-np.sin(phi_base), 0, np.cos(phi_base)]])
        self.mat1 = np.matmul(self.x_mat, self.z_mat)
        self.mat1 = np.matmul(self.y_mat, self.mat1)
        self.rmax = 1.0

    def _single_point_generator(self):
        x = np.random.uniform(0, self.theta_range), np.random.uniform(0, self.phi_range)
        return reversed([np.cos(x[0]), (np.sin(x[0])*np.sin(x[1])), (np.sin(x[0])*np.cos(x[1]))])

    def _multi_point_generator(self, num):
        x = zip(np.random.uniform(0, self.theta_range, num), np.random.uniform(0, self.phi_range, num))
        return np.array(map(lambda x : np.matmul(self.mat1, np.array([i for i in reversed([self.r*np.cos(x[0]), (self.R1)*np.sin(x[0])*np.sin(x[1]), (self.R2*np.sin(x[0])*np.cos(x[1]))])])),list(set(x)))) # ensure there are no double matches

    def _sphere_inv_metric(self, p1, p2, h):
        if np.array_equal(p1,p2) : return 0
        a = self._sphere_metric(p1, p2)
        return 1.0/a

    def _sphere_step_metric(self, p1, p2, h):
        if np.array_equal(p1,p2) : return 0
        a = self._sphere_metric(p1, p2)
        if (a < (self.rmax*h)) :
            return 1.0/a
        else :
            return 0
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

    def _cosine_metric(self, p1, p2, h):
        return (np.dot(p1, p2) + 1) / (2*h)

    def _sphere_metric(self, p1, p2):
        p1 = np.matmul(np.linalg.inv(self.y_mat),p1)
        p1 = np.matmul(np.linalg.inv(self.x_mat),p1)
        p1 = np.matmul(np.linalg.inv(self.z_mat),p1)
        p2 = np.matmul(np.linalg.inv(self.y_mat),p2)
        p2 = np.matmul(np.linalg.inv(self.x_mat),p2)
        p2 = np.matmul(np.linalg.inv(self.z_mat),p2)
        theta1 = np.arctan((np.sqrt((p1[0]/self.R2)**2 + (p1[1]/ self.R1)**2) ) / p1[2])
        phi1 = np.arctan((self.R2*p1[1])/(self.R1*p1[0]))
        if(p1[0] < 0) :
            phi1 = np.pi + phi1
        theta2 = np.arctan((np.sqrt((p2[0]/self.R2)**2 + (p2[1]/ self.R1)**2) ) / p2[2])
        phi2 = np.arctan((self.R2*p2[1])/(self.R1*p2[0]))
        if(p2[0] < 0) :
            phi2 = np.pi + phi2
        val = np.square(theta1 - theta2) / (np.pi*np.pi)
        val += np.square(phi1 - phi2) / (4*np.pi*np.pi)
        return np.sqrt(val)


    def _laplacian_matrix(self, points, h, lambda_val):
        kernel_pairwise = np.zeros([points.shape[0]]*2).astype(np.float64)
        zero = np.zeros(3)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if flags.metric == "sphere":
                    kernel_pairwise[i,j] = self._sphere_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "cosine":
                    kernel_pairwise[i,j] = self._cosine_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "sphere_inv":
                    kernel_pairwise[i,j] = self._sphere_inv_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "sphere_step":
                    kernel_pairwise[i,j] = self._sphere_step_metric(points[i] , points[j], h)/h**2
                elif flags.metric == "euclidean_step":
                    kernel_pairwise[i,j] = self._euclidean_step_metric(points[i] , points[j], h)/h**2
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
