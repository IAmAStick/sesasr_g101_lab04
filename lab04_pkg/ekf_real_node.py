
#!/usr/bin/env python3
"""Task 2 five-state EKF for a real TurtleBot3, without Nav2."""

import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from lab04_pkg.ekf_models import (
    RobotEKF,
    eval_Ht,
    eval_Ht_imu,
    eval_Ht_odom,
    eval_hx,
    eval_hx_imu,
    eval_hx_odom,
    normalize_angle,
    residual_landmark,
    residual_vector,
)

def load_tag_positions(filename):
    """Read tag_id x y into a dictionary."""
    result = {}
    path = Path(filename).expanduser()

    with path.open(encoding='utf-8') as file:
        for line_number, raw in enumerate(file, 1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f'{path}:{line_number}: expected ID X Y'
                )

            tag_id = int(fields[0])
            x, y = float(fields[1]), float(fields[2])

            if tag_id in result or not (
                math.isfinite(x) and math.isfinite(y)
            ):
                raise ValueError(
                    f'{path}:{line_number}: invalid tag data'
                )

            result[tag_id] = (x, y)

    if not result:
        raise ValueError(f'No tag positions in {path}')

    return result

def quaternion_from_yaw(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

class RealTask2EKF(Node):

    def __init__(self):
        super().__init__('ekf_real_node')

        # Topics and coordinate frames.
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('landmark_topic', '/camera/landmarks')
        self.declare_parameter('ekf_topic', '/ekf')
        self.declare_parameter('tag_positions_file', str(Path(__file__).resolve().parent / 'tag_positions.txt')) #getting tags from the same folder 
        self.declare_parameter('output_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        # Initial robot pose in the output frame.
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw_deg', 0.0)

        # Initial state uncertainty.
        self.declare_parameter('initial_std_xy', 0.5)
        self.declare_parameter('initial_std_yaw_deg', 20.0)
        self.declare_parameter('initial_std_v', 0.10)
        self.declare_parameter('initial_std_w_deg', 10.0)

        # Starting noise assumptions from the lab.
        self.declare_parameter('std_process_v', 0.03)
        self.declare_parameter('std_process_w_deg', 2.0)
        self.declare_parameter('std_odom_v', 0.05)
        self.declare_parameter('std_odom_w_deg', 5.0)
        self.declare_parameter('std_imu_w_deg', 2.0)
        self.declare_parameter('std_landmark_r', 0.10)
        self.declare_parameter(
            'std_landmark_bearing_deg', 5.0
        )

        # Hardware-specific parameters.
        self.declare_parameter('prediction_hz', 20.0)
        self.declare_parameter('max_predict_gap_s', 0.5)
        self.declare_parameter('gyro_bias_rad_s', 0.0)

        # 99% chi-square threshold for a 2D observation.
        # Setting this to zero disables landmark gating.
        self.declare_parameter('landmark_gate_chi2', 9.21) #was 9.21 to reject impossibly placed landmarks wrt prediction, but put 0.0 if causing issues w/ initial positioning

        p = lambda name: self.get_parameter(name).value
        rad = math.radians

        self.output_frame = str(p('output_frame'))
        self.base_frame = str(p('base_frame'))

        self.nominal_dt = (
            1.0 / float(p('prediction_hz'))
        )
        self.max_predict_gap = float(
            p('max_predict_gap_s')
        )
        self.gyro_bias = float(p('gyro_bias_rad_s'))
        self.landmark_gate = float(
            p('landmark_gate_chi2')
        )

        if (
            self.nominal_dt <= 0
            or self.max_predict_gap <= 0
        ):
            raise ValueError(
                'Prediction timing parameters must be positive'
            )

        # Load real tag coordinates.
        tag_file = str(
            p('tag_positions_file')
        ).strip()

        self.landmarks = (
            load_tag_positions(tag_file)
            if tag_file else {}
        )

        # TASK 2: five-dimensional state.
        # [x, y, theta, v, omega]
        self.ekf = RobotEKF(dim_x=5, dim_u=2)

        self.ekf.mu = np.array([
            float(p('initial_x')),
            float(p('initial_y')),
            normalize_angle(
                rad(float(p('initial_yaw_deg')))
            ),
            0.0,
            0.0,
        ])

        std_xy = float(p('initial_std_xy'))
        std_yaw = rad(
            float(p('initial_std_yaw_deg'))
        )
        std_v = float(p('initial_std_v'))
        std_w = rad(
            float(p('initial_std_w_deg'))
        )

        self.ekf.Sigma = np.diag([
            std_xy**2,
            std_xy**2,
            std_yaw**2,
            std_v**2,
            std_w**2,
        ])

        # Process noise covariance.
        self.process_cov_per_step = np.diag([
            float(p('std_process_v'))**2,
            rad(
                float(p('std_process_w_deg'))
            )**2,
        ])

        # Wheel odometry measurement covariance.
        self.Qt_odom = np.diag([
            float(p('std_odom_v'))**2,
            rad(
                float(p('std_odom_w_deg'))
            )**2,
        ])

        # IMU measurement covariance.
        self.Qt_imu = np.array([[
            rad(float(p('std_imu_w_deg')))**2
        ]])

        # Landmark measurement covariance.
        self.Qt_landmark = np.diag([
            float(p('std_landmark_r'))**2,
            rad(
                float(p('std_landmark_bearing_deg'))
            )**2,
        ])

        for name, matrix in (
            ('process', self.process_cov_per_step),
            ('odom', self.Qt_odom),
            ('imu', self.Qt_imu),
            ('landmark', self.Qt_landmark),
        ):
            if not np.all(np.diag(matrix) > 0):
                raise ValueError(
                    f'{name} standard deviations must be positive'
                )

        # Hardware sensor QoS.
        self.odom_sub = self.create_subscription(
            Odometry,
            str(p('odom_topic')),
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            str(p('imu_topic')),
            self.imu_callback,
            qos_profile_sensor_data,
        )

        self.landmark_sub = self.create_subscription(
            LandmarkArray,
            str(p('landmark_topic')),
            self.landmark_callback,
            qos_profile_sensor_data,
        )

        self.publisher = self.create_publisher(
            Odometry,
            str(p('ekf_topic')),
            10,
        )

        self.have_odom = False
        self.last_prediction_ns = None
        self.last_frame_warning = False
        self.rejected_landmarks = 0

        self.timer = self.create_timer(
            self.nominal_dt,
            self.predict_callback,
        )

        self.get_logger().info(
            f'Task 2 EKF started. '
            f'Known tags: {len(self.landmarks)}. '
            f'Frame: {self.output_frame}.'
        )

        if not self.landmarks:
            self.get_logger().warn(
                'No tag file supplied. Running '
                'odom + IMU fusion without landmarks.'
            )
        else:
            self.get_logger().warn(
                'Initial pose must be in the tag-map frame. '
                'No automatic tag initialization or TF publishing.'
            )

    # --------------------------------------------------
    # Generic measurement update
    # --------------------------------------------------

    def _update(self, z, hx, jac, covariance):
        self.ekf.update(
            z=z,
            eval_hx=hx,
            eval_Ht=jac,
            Qt=covariance,
            Ht_args=(self.ekf.mu,),
            hx_args=(self.ekf.mu,),
            residual=residual_vector,
        )

    # --------------------------------------------------
    # Wheel odometry: correct v and omega
    # --------------------------------------------------

    def odom_callback(self, msg):

        z = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        ], dtype=float)

        if not np.all(np.isfinite(z)):
            return

        if (
            not self.last_frame_warning
            and msg.child_frame_id
        ):
            child = msg.child_frame_id.lstrip('/')

            if child not in (
                self.base_frame.lstrip('/'),
                'base_link',
                'base_footprint',
            ):
                self.get_logger().warn(
                    f'/odom twist frame is '
                    f'{msg.child_frame_id!r}; '
                    'check frame conversion.'
                )

            self.last_frame_warning = True

        self._update(
            z,
            eval_hx_odom,
            eval_Ht_odom,
            self.Qt_odom,
        )

        if not self.have_odom:
            self.have_odom = True
            self.last_prediction_ns = (
                self.get_clock().now().nanoseconds
            )

    # --------------------------------------------------
    # IMU: correct omega
    # --------------------------------------------------

    def imu_callback(self, msg):

        # Assumes IMU +Z is aligned with robot +Z.
        # Gyroscope angular velocity is in rad/s.

        if msg.angular_velocity_covariance[0] == -1.0:
            return

        measured_w = float(
            msg.angular_velocity.z
        )

        if not math.isfinite(measured_w):
            return

        z = np.array([
            measured_w - self.gyro_bias
        ])

        self._update(
            z,
            eval_hx_imu,
            eval_Ht_imu,
            self.Qt_imu,
        )

    # --------------------------------------------------
    # Landmark correction: x, y, theta
    # --------------------------------------------------

    def landmark_callback(self, msg):

        if not self.landmarks:
            return

        updated = False

        for landmark in msg.landmarks:

            tag_id = int(landmark.id)

            if tag_id not in self.landmarks:
                continue

            z = np.array([
                landmark.range,
                landmark.bearing,
            ], dtype=float)

            if (
                not np.all(np.isfinite(z))
                or z[0] <= 0
            ):
                continue

            tag_xy = self.landmarks[tag_id]

            # Optional innovation gating.
            if self.landmark_gate > 0:

                H = eval_Ht(
                    self.ekf.mu,
                    tag_xy,
                )

                expected = eval_hx(
                    self.ekf.mu,
                    tag_xy,
                )

                innovation = residual_landmark(
                    z,
                    expected,
                )

                S = (
                    H @ self.ekf.Sigma @ H.T
                    + self.Qt_landmark
                )

                d2 = float(
                    innovation
                    @ np.linalg.solve(S, innovation)
                )

                if d2 > self.landmark_gate:
                    self.rejected_landmarks += 1

                    if self.rejected_landmarks <= 3:
                        self.get_logger().warn(
                            f'Rejected tag {tag_id}, '
                            f'd^2={d2:.2f}. '
                            'Check initial pose, '
                            'tag positions and camera TF.'
                        )

                    continue

            self.ekf.update(
                z=z,
                eval_hx=eval_hx,
                eval_Ht=eval_Ht,
                Qt=self.Qt_landmark,
                Ht_args=(self.ekf.mu, tag_xy),
                hx_args=(self.ekf.mu, tag_xy),
            )

            updated = True

        if updated:
            self.publish_estimate()

    # --------------------------------------------------
    # Prediction using real elapsed time
    # --------------------------------------------------

    def predict_callback(self):

        if not self.have_odom:
            return

        now_ns = self.get_clock().now().nanoseconds

        if self.last_prediction_ns is None:
            self.last_prediction_ns = now_ns
            return

        dt = (
            now_ns - self.last_prediction_ns
        ) * 1e-9

        self.last_prediction_ns = now_ns

        if dt <= 0:
            return

        if dt > self.max_predict_gap:
            self.get_logger().warn(
                f'Prediction gap {dt:.2f}s; '
                'skipping this integration.'
            )
            return

        # Subdivide long timer intervals.
        steps = max(
            1,
            math.ceil(dt / self.nominal_dt),
        )

        step_dt = dt / steps

        for _ in range(steps):

            self.ekf.Mt = (
                self.process_cov_per_step
                * (step_dt / self.nominal_dt)
            )

            self.ekf.predict(
                g_extra_args=(step_dt,)
            )

        self.publish_estimate()

    # --------------------------------------------------
    # Publish nav_msgs/Odometry
    # --------------------------------------------------

    def publish_estimate(self):

        msg = Odometry()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = self.output_frame
        msg.child_frame_id = self.base_frame

        msg.pose.pose.position.x = float(
            self.ekf.mu[0]
        )
        msg.pose.pose.position.y = float(
            self.ekf.mu[1]
        )
        msg.pose.pose.orientation = (
            quaternion_from_yaw(
                float(self.ekf.mu[2])
            )
        )

        msg.twist.twist.linear.x = float(
            self.ekf.mu[3]
        )
        msg.twist.twist.angular.z = float(
            self.ekf.mu[4]
        )

        # Pose covariance mapping:
        # EKF [x,y,theta] -> ROS [x,y,z,roll,pitch,yaw]

        pose_cov = np.zeros((6, 6))

        for i in (2, 3, 4):
            pose_cov[i, i] = 1e6

        pose_idx = [0, 1, 5]

        pose_cov[
            np.ix_(pose_idx, pose_idx)
        ] = self.ekf.Sigma[:3, :3]

        msg.pose.covariance = (
            pose_cov.ravel().tolist()
        )

        # Twist covariance mapping:
        # EKF [v,omega] -> ROS [vx,vy,vz,wx,wy,wz]

        twist_cov = np.zeros((6, 6))

        for i in (1, 2, 3, 4):
            twist_cov[i, i] = 1e6

        twist_idx = [0, 5]

        twist_cov[
            np.ix_(twist_idx, twist_idx)
        ] = self.ekf.Sigma[3:5, 3:5]

        msg.twist.covariance = (
            twist_cov.ravel().tolist()
        )

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = RealTask2EKF()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()
