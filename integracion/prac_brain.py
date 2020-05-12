from pyrobot.brain import Brain

import math
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pygame
import os
import prac_run as segment

class pracbrain(Brain):

  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  def callback(self,data):
    self.rosImage = data

  def setup(self):
    self.image_sub = rospy.Subscriber("/image",Image,self.callback)
    self.bridge = CvBridge()
    self.seg = segment.Segmentador()
    print os.getcwd()
    if os.path.exists('/home/robotica/roboticaUPM-master/clasificadores/segmentacion.pkl'):
            self.seg.clf_load()
    else:
            self.seg.clf_create('/home/robotica/roboticaUPM-master/rsc/imgs_seg', '/home/robotica/roboticaUPM-master/rsc/imgsMk_seg')


  def step(self):
    try:
      self.cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Stage Camera Image", self.cv_image)
    cv2.waitKey(1)

    tv, fv = self.seg.video_create(self.cv_image)
    self.move(fv,tv)
    

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    return pracbrain('BrainFinalExam', engine)
