from geometry_msgs.msg import Pose, PoseArray, Quaternion, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.cluster.hierarchy import linkage, fcluster, fclusterdata
from . util import rotateQuaternion, getHeading
from . pf_base import PFLocaliserBase
from time import time
import numpy as np
import random
import rospy
import math
import copy

class PFLocaliser(PFLocaliserBase):
    
    # ----- Initialization parameter
    def __init__(self):

        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters

        self.ROTATION_NOISE = np.random.uniform(0.01, 0.3)
        self.TRANSLATION_NOISE = np.random.uniform(0.01, 0.3)
        self.DRIFT_NOISE = np.random.uniform(0.01, 0.3)

        # ----- Set sensor model parameters

        self.NUMBER_PREDICTED_READINGS = 90
        self.PARTICLE_COUNT = 200

        # ----- Set noise parameters

        self.PARTICLE_POS_NOISE = np.random.uniform(75, 100)
        self.PARTICLE_ANG_NOISE = np.random.uniform(1, 120)
        
    def cloud_converge(self, poses):
        noise = []
        noise.append(self.ROTATION_NOISE)
        noise.append(self.TRANSLATION_NOISE)
        noise.append(self.DRIFT_NOISE)
        noise.append(self.PARTICLE_POS_NOISE)
        noise.append(self.PARTICLE_ANG_NOISE)
        noise[0] = noise[0] - 0.01 if noise[0]> 0.1 else np.random.uniform(0.01, 0.1) 
        noise[1] = noise[1] - 0.01 if noise[1]> 0.1 else np.random.uniform(0.01, 0.1) 
        noise[2] = noise[2] - 0.01 if noise[2]> 0.1 else np.random.uniform(0.01, 0.1) 
        noise[3] = noise[3] - 0.1 if noise[2]> 2.0 else np.random.uniform(0.01, 2) 
        noise[4] = noise[4] - 1.0 if noise[2]> 90 else np.random.uniform(0.01, 90) 
        poses.position.x += random.gauss(0, noise[3]) * noise[1]
        poses.position.y += random.gauss(0, noise[3]) * noise[2]
        angular_noise = (random.vonmisesvariate(0, noise[4]) - math.pi) * noise[0]
        poses.orientation = rotateQuaternion(poses.orientation, angular_noise)

        return poses

    def initialise_particle_cloud(self, initial_pose):
        """
        Called whenever an initial_pose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        
        :Args:
            | initial_pose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        pose_array = PoseArray()

        for _ in range(self.PARTICLE_COUNT):
            pose = Pose()

            positional_noise_x = random.gauss(0, self.PARTICLE_POS_NOISE) * self.TRANSLATION_NOISE
            positional_noise_y = random.gauss(0, self.PARTICLE_POS_NOISE) * self.DRIFT_NOISE

            pose.position.x = initial_pose.pose.pose.position.x + positional_noise_x
            pose.position.y = initial_pose.pose.pose.position.y + positional_noise_y

            angular_noise = (random.vonmisesvariate(0, self.PARTICLE_ANG_NOISE) - math.pi) * self.ROTATION_NOISE

            pose.orientation = rotateQuaternion(initial_pose.pose.pose.orientation, angular_noise)

            pose_array.poses.append(pose)
            
        return pose_array

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. 
        
        i.e. self.particlecloud should be updated
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update
        """
        
        global latest_scan
        latest_scan = scan

        # Re-sample the particle cloud by creating a particle with a new position and orientation
        # based on the probability of the particle
        # Roulette algorithm

        weights = []
        cumulative_weight = 0

        for poses in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, poses)
            weights.append(weight)
            cumulative_weight += weight

        pose_array = PoseArray()

        for _ in range(len(self.particlecloud.poses)):
            random_threshold = random.random() * cumulative_weight

            weight_sum = 0
            i = 0

            while weight_sum < random_threshold:
                weight_sum += weights[i]
                i += 1

            pose_array.poses.append(copy.deepcopy(self.particlecloud.poses[i - 1]))

        # In order to converge to the new pose, adjust the noise parameters         
        
        for poses in pose_array.poses:
            poses = self.cloud_converge(poses) 
            
        self.particlecloud = pose_array

        # Output the estimated position and orientation of the robot

        robot_estimated_position = self.estimatedpose.pose.pose.position
        robot_estimated_orientation = self.estimatedpose.pose.pose.orientation
    
        # Transformation

        robot_quaternion = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(robot_quaternion)

        # Output the position and orientation estimates of the robot

        print("/Robot estimated Pose: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
            x=robot_estimated_position.x, y=robot_estimated_position.y, yaw=math.degrees(yaw)))

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """

        """
        # Mean pose
        estimated_pose = Pose() 

        position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))
        
        for poses in self.particlecloud.poses:
            position_sum_x += poses.position.x 
            position_sum_y += poses.position.y 
            orientation_sum_z += poses.orientation.z 
            orientation_sum_w += poses.orientation.w 

        estimated_pose.position.x = position_sum_x    / self.PARTICLE_COUNT 
        estimated_pose.position.y = position_sum_y    / self.PARTICLE_COUNT 
        estimated_pose.orientation.z = orientation_sum_z / self.PARTICLE_COUNT 
        estimated_pose.orientation.w = orientation_sum_w / self.PARTICLE_COUNT    
        """
        # Hierarchical agglomerate clustering

        # Set the clustering distance threshold

        cluster_distance_threshold = 0.3
        estimated_pose = Pose()

        # Initialize position and direction array variables

        position_x, position_y, orientation_z, orientation_w = ([] for _ in range(4))

        for pose in self.particlecloud.poses:
            position_x.append(pose.position.x)
            position_y.append(pose.position.y)
            orientation_z.append(pose.orientation.z)
            orientation_w.append(pose.orientation.w)

        position_x = np.array(position_x)
        position_y = np.array(position_y)
        orientation_z = np.array(orientation_z)
        orientation_w = np.array(orientation_w)

        # form a distance matrix

        dis_matrix = np.column_stack((position_x, position_y, orientation_z, orientation_w))

        link_matrix = linkage(dis_matrix, method='median')

        cluster_characteristics = fcluster(link_matrix, cluster_distance_threshold, criterion='distance')

        cluster_count = max(cluster_characteristics)
        cluster_counts = [0] * cluster_count
        cluster_weight_sums = [0] * cluster_count

        for i, cluster_characteristic in enumerate(cluster_characteristics):
            pose = self.particlecloud.poses[i]

            cluster_weight = self.sensor_model.get_weight(latest_scan, pose)

            cluster_counts[cluster_characteristic - 1] += 1
            cluster_weight_sums[cluster_characteristic - 1] += cluster_weight

        # find the most accurate clusters of particles overall

        cluster_highest_weight = cluster_weight_sums.index(max(cluster_weight_sums)) + 1
        cluster_highest_weight_count = cluster_counts[cluster_highest_weight - 1]

        # Initializes the position and orientation

        position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))
        for i, cluster_characteristic in enumerate(cluster_characteristics):
            if cluster_characteristic == cluster_highest_weight:
                pose = self.particlecloud.poses[i]
                position_sum_x += pose.position.x
                position_sum_y += pose.position.y
                orientation_sum_z += pose.orientation.z
                orientation_sum_w += pose.orientation.w

        estimated_pose.position.x = position_sum_x / cluster_highest_weight_count
        estimated_pose.position.y = position_sum_y / cluster_highest_weight_count
        estimated_pose.orientation.z = orientation_sum_z / cluster_highest_weight_count
        estimated_pose.orientation.w = orientation_sum_w / cluster_highest_weight_count

        # Output the estimated position and orientation of the robot

        robot_estimated_position = estimated_pose.position
        robot_estimated_orientation = estimated_pose.orientation

        # Transformation

        robot_quaternion = [robot_estimated_orientation.x, robot_estimated_orientation.y,
                            robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(robot_quaternion)

        # Output the position and orientation estimates of the robot
        print("Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
            x=robot_estimated_position.x, y=robot_estimated_position.y, yaw=math.degrees(yaw)))

        return estimated_pose
