from pyrobot.brain import Brain

import math

class BrainTestNavigator(Brain):
 
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
    pass

  def step(self):
    hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
    print "I got from the simulation",hasLine,lineDistance,searchRange

    if (hasLine):
      if (lineDistance > self.NO_ERROR):
        self.move(self.FULL_FORWARD,self.HARD_LEFT)
      elif (lineDistance < self.NO_ERROR):
        self.move(self.FULL_FORWARD,self.HARD_RIGHT)
      else:
        self.move(self.FULL_FORWARD,self.NO_TURN)
    else:
      # if we can't find the line we just stop, this isn't very smart
      self.move(self.NO_FORWARD,self.NO_TURN)
 
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
