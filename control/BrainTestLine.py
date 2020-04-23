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
    pass

  def step(self):
    hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
    print "I got from the simulation",hasLine,lineDistance,searchRange

    

    if hasLine:
      error = lineDistance - self.prev_error
      self.prev_error = lineDistance
      tv = (lineDistance/searchRange) - math.copysign(1,error)*error*0.2
      fv = max(0, 1 - abs(tv*1.5))
      self.move(fv, tv)
      
    elif not hasLine and self.robot.range[5] < 0.5 or self.robot.range[6] < 0.5:
      self.move(self.FULL_FORWARD, self.MED_LEFT)

    else:
      # if we can't find the line we just stop, this isn't very smart
      self.move(self.NO_FORWARD, math.copysign(1,self.prev_error)*0.3)
 
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
