from pyrobot.brain import Brain

import math
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pygame
import segmentacion.prac_run as segment
import reconocimiento.reconocimiento as recon
import os


class BrainFinalExam(Brain):

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


    def setup(self):
        self.object = False
        self.fv = 0
        self.tv = 0
        self.image_sub = rospy.Subscriber("/image", Image, self.callback)
        self.bridge = CvBridge()
        self.seg = segment.Segmentador()
        if os.path.exists('/home/robotica/roboticaUPM/clasificadores/segmentacion.pkl'):
            self.seg.clf_load()
        else:
            self.seg.clf_create("/home/robotica/roboticaUPM/rsc/imgs_seg/", "/home/robotica/roboticaUPM/rsc/imgsMk_seg/")
        self.rec = recon.Reconocimiento()
        if os.path.exists('/home/robotica/roboticaUPM/clasificadores/reconomientoORB.pkl'):
            self.rec.clf_load_orb()
        else:
            self.rec.clf_create_orb("/home/robotica/roboticaUPM/rsc/marcas/")


    def callback(self, data):
        self.rosImage = data


    def destroy(self):
        cv2.destroyAllWindows()


    def step(self):

        if self.stop:
            return
        # take the last image received from the camera and convert it into
        # opencv format
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
            # self.cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "rgb8")
        except CvBridgeError as e:
            print(e)

        #    # use pygame to display the image
        #surface = pygame.display.set_mode(self.cv_image.shape[-2::-1])
        #pygimage = pygame.image.frombuffer(cv2.cvtColor(self.cv_image,
        #cv2.COLOR_BGR2RGB),
        #self.cv_image.shape[-2::-1], 'RGB')
        #surface.blit(pygimage, (0, 0))
        #pygame.display.flip() # update the display

        # write the image to a file, for debugging etc.
        # cv2.imwrite("test-file.jpg",self.cv_image)

        # Here you should process the image from the camera and calculate
        # your control variable(s), for now we will just give the controller
        # some 'fixed' values so that it will do something.

        # A trivial on-off controller
        hasLine, self.tv, self.fv = self.seg.analisis(self.cv_image)

        fig = self.rec.analisis(self.cv_image, self.seg.ultSalida)
        if fig in [0, 1, 2, 3]:
            print('Identificado ORB {} '.format(self.rec.etiquetas[fig[0]]))
            # cv2.putText(self.cv_image, 'Identificado ORB {} '.format(self.rec.etiquetas[fig[0]]), (15, 40),
            #            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))


        # display the image using opencv
        cv2.imshow("Stage Camera Image", self.cv_image)
        cv2.waitKey(1)

        # Para rodear un objeto
        frente = [self.robot.range[3].distance(),
                  self.robot.range[4].distance()]
        if min(*frente) < 0.35:
            self.object = True
            self.move(0, 1)
            return

        if self.object:
            if self.robot.range[5].distance() < 0.5:
                self.fv = 0
                self.tv = 0.5
            elif self.robot.range[6].distance() < 0.5:
                self.fv = 0.4
                self.tv = 0.4
            elif self.robot.range[7].distance() < 0.5:
                self.fv = 0.4
                self.tv = -0.5
                self.object = not hasLine

            elif self.robot.range[7].distance() > 0.5 and self.robot.range[7].distance() < 0.8:
                self.fv = 0.3
                self.tv = -0.3

        self.move(self.fv, self.tv)
        return


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    return BrainFinalExam('BrainFinalExam', engine)
