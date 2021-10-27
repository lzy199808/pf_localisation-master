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
        self.ROTATION_NOISE = np.random.uniform(0.05, 0.2)
        self.TRANSLATION_NOISE = np.random.uniform(0.05, 0.2)
        self.DRIFT_NOISE = np.random.uniform(0.05, 0.2)

        # ----- Set sensor model parameters
        self.PARTICLE_COUNT = 200
        self.NUMBER_PREDICTED_READINGS = 20

        # ----- Set noise parameters
        self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(75, 100)
        self.PARTICLE_ANGULAR_NOISE = np.random.uniform(1, 120)


  
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

            positional_noise_x = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.TRANSLATION_NOISE
            positional_noise_y = random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.DRIFT_NOISE

            pose.position.x = initial_pose.pose.pose.position.x + positional_noise_x
            pose.position.y = initial_pose.pose.pose.position.y + positional_noise_y

            ANGULAR_NOISE = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ROTATION_NOISE

            pose.orientation = rotateQuaternion(initial_pose.pose.pose.orientation, ANGULAR_NOISE)

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
            if self.ROTATION_NOISE > 0.2:
                self.ROTATION_NOISE -= 0.05

            else:
                self.ROTATION_NOISE = np.random.uniform(0.05, 0.2)

            if self.TRANSLATION_NOISE > 0.2:
                self.TRANSLATION_NOISE -= 0.05

            else:
                self.TRANSLATION_NOISE = np.random.uniform(0.05, 0.2)

            if self.DRIFT_NOISE > 0.2:
                self.DRIFT_NOISE -= 0.05

            else:
                self.DRIFT_NOISE = np.random.uniform(0.05, 0.2)

            if self.PARTICLE_POSITIONAL_NOISE > 2.0:
                self.PARTICLE_POSITIONAL_NOISE -= 0.1

            else:
                self.PARTICLE_POSITIONAL_NOISE = np.random.uniform(0.05, 2)

            if self.PARTICLE_ANGULAR_NOISE > 90.0:
                self.PARTICLE_ANGULAR_NOISE -= 1.0

            else:
                self.PARTICLE_ANGULAR_NOISE = np.random.uniform(0.05, 90)

            # Adds position noise to the x and y coordinates
            poses.position.x += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.TRANSLATION_NOISE
            poses.position.y += random.gauss(0, self.PARTICLE_POSITIONAL_NOISE) * self.DRIFT_NOISE

            # Add Angular noise
            ANGULAR_NOISE = (random.vonmisesvariate(0, self.PARTICLE_ANGULAR_NOISE) - math.pi) * self.ROTATION_NOISE

            # Add orientation noise
            poses.orientation = rotateQuaternion(poses.orientation, ANGULAR_NOISE)
        
        self.particlecloud = pose_array
        """
        Output the estimated position and orientation of the robot
        """
        Robot_estimated_position = self.estimatedpose.pose.pose.position
        Robot_estimated_orientation = self.estimatedpose.pose.pose.orientation
    
        # Transformation
        Robot_Quaternion = [Robot_estimated_orientation.x, Robot_estimated_orientation.y, Robot_estimated_orientation.z, Robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(Robot_Quaternion)

        # Output the position and orientation estimates of the robot
        print("/Robot estimated Pose: Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    x=Robot_estimated_position.x,
                                                                                                    y=Robot_estimated_position.y,
                                                                                                    yaw=math.degrees(yaw)))


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """
        # Set the clustering distance threshold
        CLUSTER_DISTANCE_THRESHOLD = 0.3
        """
        # Mean pose
        estimated_pose = Pose() 

        position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))
        
        for pose_object in self.particlecloud.poses:
            position_sum_x    += pose_object.position.x 
            position_sum_y    += pose_object.position.y 
            orientation_sum_z += pose_object.orientation.z 
            orientation_sum_w += pose_object.orientation.w 

        estimated_pose.position.x    = position_sum_x    / self.PARTICLE_COUNT 
        estimated_pose.position.y    = position_sum_y    / self.PARTICLE_COUNT 
        estimated_pose.orientation.z = orientation_sum_z / self.PARTICLE_COUNT 
        estimated_pose.orientation.w = orientation_sum_w / self.PARTICLE_COUNT    
        """
        # Hierarchical agglomerate clustering
        estimated_pose = Pose()

        # Initialize position and direction array variables
        position_x, position_y, orientation_z, orientation_w = ([] for _ in range(4))

        for pose in self.particlecloud.poses:
            position_x.append(pose.position.x)
            position_y.append(pose.position.y)
            orientation_z.append(pose.orientation.z)
            orientation_w.append(pose.orientation.w)

        position_x    = np.array(position_x)
        position_y    = np.array(position_y)
        orientation_z = np.array(orientation_z)
        orientation_w = np.array(orientation_w)

        # form a distance matrix
        distance_matrix = np.column_stack((position_x, position_y, orientation_z, orientation_w))

        linkage_matrix = linkage(distance_matrix, method='median')

        particle_cluster_identities = fcluster(linkage_matrix, CLUSTER_DISTANCE_THRESHOLD, criterion='distance')
        #print(particle_cluster_identities)

        cluster_count = max(particle_cluster_identities)
        cluster_particle_counts = [0] * cluster_count
        cluster_probability_weight_sums = [0] * cluster_count

        for i, particle_cluster_identity in enumerate(particle_cluster_identities):
            pose = self.particlecloud.poses[i]

            probability_weight = self.sensor_model.get_weight(latest_scan, pose)

            cluster_particle_counts[particle_cluster_identity - 1] += 1
            cluster_probability_weight_sums[particle_cluster_identity - 1] += probability_weight

        # find the most accurate clusters of particles overall
        cluster_highest_belief = cluster_probability_weight_sums.index(max(cluster_probability_weight_sums)) + 1
        cluster_highest_belief_particle_count = cluster_particle_counts[cluster_highest_belief - 1]

        # Initializes the position and orientation
        position_sum_x, position_sum_y, orientation_sum_z, orientation_sum_w = (0 for _ in range(4))

        for i, particle_cluster_identity in enumerate(particle_cluster_identities):
            if (particle_cluster_identity == cluster_highest_belief):
                pose = self.particlecloud.poses[i]

                position_sum_x    += pose.position.x
                position_sum_y    += pose.position.y
                orientation_sum_z += pose.orientation.z
                orientation_sum_w += pose.orientation.w

        estimated_pose.position.x    = position_sum_x    / cluster_highest_belief_particle_count
        estimated_pose.position.y    = position_sum_y    / cluster_highest_belief_particle_count
        estimated_pose.orientation.z = orientation_sum_z / cluster_highest_belief_particle_count
        estimated_pose.orientation.w = orientation_sum_w / cluster_highest_belief_particle_count

        """
        Output the estimated position and orientation of the robot
        """
        robot_estimated_position = estimated_pose.position
        robot_estimated_orientation = estimated_pose.orientation

        # Transformation
        Robot_Quaternion = [robot_estimated_orientation.x, robot_estimated_orientation.y, robot_estimated_orientation.z, robot_estimated_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(Robot_Quaternion)

        # Output the position and orientation estimates of the robot
        print("Robot Position: [{x:.2f}, {y:.2f}]    Robot Orientation: [{yaw:.2f}]".format(
                                                                                                    x=robot_estimated_position.x, 
                                                                                                    y=robot_estimated_position.y, 
                                                                                                    yaw=math.degrees(yaw)))

        return estimated_pose
