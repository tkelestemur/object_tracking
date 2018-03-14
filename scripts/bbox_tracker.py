#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from image_geometry import PinholeCameraModel

import cv2
import dlib
import numpy as np
from math import isnan

class BoundingBoxTracker():

    def __init__(self):

        rospy.init_node('bbox_tracker')

        rgb_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
        camera_info_topic = "/hsrb/head_rgbd_sensor/depth_registered/camera_info"
        depth_topic = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"

        self.rgb_sub = rospy.Subscriber(rgb_topic, Image, self.rgb_cb)
        self.rgb_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.rgb_info_cb)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_cb)
        self.tracking_rgb_pub = rospy.Publisher("/object_tracking/bbox_image", Image,  queue_size=10)

        self.roi_selected = False
        self.cv_bridge = CvBridge()
        self.tracking_rate = 20
        self.first_image = False
        self.first_image_info = False

        self.cam_model = PinholeCameraModel()

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CORR']
        tracker_type = tracker_types[6]

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'CORR':
            self.tracker = dlib.correlation_tracker()

        rospy.sleep(1) # settle down

    def rgb_cb(self, msg):

        try:
            self.frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.first_image = True

        except CvBridgeError, e:
            a=e

    def rgb_info_cb(self, msg):
        self.cam_info = msg
        self.first_image_info = True


    def depth_cb(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.depth_image = np.squeeze(np.array(image, dtype=np.float32))

    def track(self):
        while not self.first_image or not self.first_image_info:
            rospy.loginfo('waiting for the first image!')
            rospy.sleep(0.2)

        self.cam_model.fromCameraInfo(self.cam_info)

        rate = rospy.Rate(self.tracking_rate)
        rospy.loginfo('select region of interest!')

        self.roi = cv2.selectROI('roi_selection', self.frame, False)
        cv2.destroyWindow('roi_selection')
        self.roi_selected = True
        rospy.loginfo(self.roi)

        rect = dlib.rectangle(self.roi[0], self.roi[1], (self.roi[0]+self.roi[2]), (self.roi[1]+self.roi[3]))
        self.tracker.start_track(self.frame, rect) # init dlib tracker

        while not rospy.is_shutdown():

            tracking_quality = self.tracker.update(self.frame)
            rospy.loginfo('tracking_quality: ' + str(tracking_quality))
            if tracking_quality >= 3.0:
                bbox =  self.tracker.get_position()

                p1 = (int(bbox.left()), int(bbox.top()))
                p2 = (int(bbox.right()), int(bbox.bottom()))
                cv2.rectangle(self.frame, p1, p2, (255,0,0), 3, 1)

                bbox_center = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
                center_z = self.depth_image[bbox_center[1], bbox_center[0]]

                if self._validate_depth_point(center_z):
                    [vx,vy,vz] = self.cam_model.projectPixelTo3dRay(bbox_center)

                    center_z = center_z / 1000
                    center_x = vx * center_z
                    center_y = vy * center_z

                    rospy.loginfo('boundix box: ' + str(p1) + str(p2))
                    rospy.loginfo('boundix box center: ' + str(bbox_center))
                    rospy.loginfo('x y z: ' + str(center_x) + ' ' + str(center_y) + ' ' + str(center_z))

                    cv2.circle(self.frame,bbox_center, 5, (0,0,255), -1)
                    # cv2.imshow('image', self.frame)
                    # cv2.waitKey(3)

                    msg = self.cv_bridge.cv2_to_imgmsg(self.frame)
                    self.tracking_rgb_pub.publish(msg)
                else:
                    rospy.logwarn('depth point is not valid!')
            rate.sleep()
    def _validate_depth_point(self, point):
        if isnan(point) or point == 0:
            return False
        return True

if __name__ == '__main__':

    tracker = BoundingBoxTracker()
    tracker.track()
    # rospy.spin()
