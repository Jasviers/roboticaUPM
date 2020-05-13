import math

from pyrobot.brain import Brain


class BrainTestNavigator(Brain):

    prev_error = 0
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

    def step(self):
        hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
        print("I got from the simulation", hasLine, lineDistance, searchRange)

        if min([i.distance() for i in self.robot.range]) < 0.5:
            self.object = True

        if self.object:
            if self.robot.range[5].distance() < 0.5:
                self.fv = 0
                self.tv = 0.5
            elif self.robot.range[6].distance() < 0.5:
                self.fv = 0.4
                self.tv = 0.4
            elif self.robot.range[7].distance() < 0.5:
                if hasLine:
                    self.object = False
                self.fv = 0.4
                self.tv = -0.5
            elif self.robot.range[7].distance() > 0.5 and self.robot.range[7].distance() < 0.8:
                self.fv = 0.3
                self.tv = -0.3

        if hasLine and not self.object:
            error = lineDistance - self.prev_error
            self.prev_error = lineDistance
            self.tv = (lineDistance/searchRange) - math.copysign(1,error)*error*0.2
            self.fv = max(0, 1 - abs(self.tv*1.5))

        elif not hasLine and not self.object:
            self.fv = 0
            self.tv = self.prev_error / searchRange

        self.move(self.fv, self.tv)

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    return BrainTestNavigator('BrainTestNavigator', engine)
