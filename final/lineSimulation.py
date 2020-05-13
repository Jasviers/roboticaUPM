"""
A wrapper for the TkSimulator to add markers to the simulator
"""

import math
import sys
from pyrobot.simulators.pysim import TkSimulator
from PIL import Image, ImageTk
import numpy as np

class LineSimulation(TkSimulator):

   LINE_SEACH_RADIUS = 10 
   BLACK_THRESHOLD = 185 # needs to be less that this for there to be line
   ROBOT_DIAMETER = 14

   def __init__(self, dimensions, offsets, scale, root = None, run = 1,
                background=None):
        TkSimulator.__init__(self, dimensions=dimensions,
	                     offsets=offsets, scale=scale, root=root,
			     run=run)
	self.backgroundFile = background
	if (self.backgroundFile != None):
          try:
            self.backgroundImage = ImageTk.PhotoImage(Image.open(background),
                                                      master=self)
	    self.canvas.create_image(0,0, anchor="nw", 
				     image=self.backgroundImage,
				     tags="background")
          except:
            self.backgroundImage = None
            print "Error opening image file ",background," I will ignore it."
	    print "",sys.exc_info()[0]
	    print "",sys.exc_info()[1]
	    print "",sys.exc_info()[2]

   def redraw(self):
      TkSimulator.redraw(self)

      # if we have a background image, redraw it as well.
      try: 
        if (self.backgroundImage != None):
	  self.canvas.create_image(0,0, anchor="nw", 
				  image=self.backgroundImage,
				  tags="background")
          self.canvas.lower("background","line")
      except:
        pass

   def getLineProperties(self): 
      # for now we ASSUME that there is only one robot!!
      robot = self.robots[0]
      #print "robot", robot._gx, robot._gy, robot._ga

      return self.symLine(math.trunc(self.scale_y(robot._gy)), 
                          math.trunc(self.scale_x(robot._gx)),
			  robot._ga, 
			  self.LINE_SEACH_RADIUS,
			  np.array(Image.open(self.backgroundFile)))

   def bilInter(self, x, y, points):
       '''Interpolate (x,y) from values associated with four points.
       The four points are a list of four triplets:  (x, y, value).  The
       four points can be in any order.  They should form a rectangle.
   
           >>> bilinear_interpolation(12, 5.5,
           ...                        [((10, 4), 100),
           ...                         ((20, 4), 200),
           ...                         ((10, 6), 150),
           ...                         ((20, 6), 300)])
           165.0
   
       '''
       ((x1,y1),q11),((_x1,y2),q12),((x2,_y1),q21),((_x2,_y2),q22) = points
   
       if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
           raise ValueError('points do not form a rectangle')
       if not x1 <= x <= x2 or not y1 <= y <= y2:
           raise ValueError('(x, y) not within the rectangle')
   
       return (q11 * (x2 - x) * (y2 - y) +
               q21 * (x - x1) * (y2 - y) +
               q12 * (x2 - x) * (y - y1) +
               q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)
   
   def fourNeigh(self,x,y):
       '''Returns the four neighbours of point (x,y) ordered by x then by y '''
       return [(math.trunc(x),math.trunc(y)),(math.trunc(x),math.trunc(y)+1),
               (math.trunc(x)+1,math.trunc(y)),
	       (math.trunc(x)+1,math.trunc(y)+1)]
   
   def pixValBi(self,(x,y),img):
       ''' Returns the value of position (x,y) in the image using
           bilinear interpolation from its four neighbours
       '''
       neig=self.fourNeigh(x,y)
       neig_vals=[img[i] for i in neig]
       return self.bilInter(x,y,zip(neig,neig_vals))
   
   def symLine(self,x,y,phi,delta,img):
       '''Simulates a line-finding algorithm, it assumes that AT MOST
          ONE line exists in the search area.

          Receives:
	    x,y: co-ordinates of the center position of the robot (pixels)
            phi: orientation of the robot, as used by pyrorobot.
            delta: length of half the image search area (in pixels)
            img: numpy array representing the image where pixels are sampled
	  returns:
            ret_val: True if a line is found, False otherwise
	    centroid: The distance from the center of the search area,
	      negative is to the left, positive to the right
	    delta: half the width of the search area, centriod is in the
	      range -delta and delta
	   
       '''

       # If given a color image, first convert it to gray
       if len(img.shape) == 3:
           #discard dimensions higher than 3 ....
           img=np.dot(img[...,:3], [0.3, 0.6, 0.1])

       # we want the "search line" to be perpindicular to the direction
       # of the robot, so we rotate the robot's direction 90 degrees to
       # get the "search line"'s direction
       phi = phi-(math.pi/2)

       # calculate the center of the "search line", to place the line in
       # front of the robot we add the diameter of the robot to the
       # position of the line
       p0=np.array([x+(math.sin(phi)*self.ROBOT_DIAMETER),
                    y-(math.cos(phi)*self.ROBOT_DIAMETER)])

       # direction of the "search line" in matrix form
       v=np.array([math.cos(phi),math.sin(phi)])

       # the points on the 'search line" to be observed
       ptsl=[tuple(p0+(inc*v)) for inc in range(-delta,delta+1)]

       # Remove points close to the borders of picture, this also
       # prevents out of range errors when the search area falls outside
       # the image
       maxX,maxY=img.shape[:2]
       weights=range(-delta,delta+1)
       pts=np.array([(x,y) for x,y in ptsl if ((x > 1) and (y > 1) and 
                                              (x < maxX-2) and (y < maxY-2))])
       weights=np.array([w for (w,(x,y)) in zip(weights,ptsl) if ((x > 1)
                        and (y > 1) and (x < maxX-2) and (y < maxY-2))])

       ret_val=False
       centroid=0

       # it is possible that the robot is so close to the border of the
       # image that there are NO pixels in the 'search line', if that is
       # the case we return a fail.

       if (len(pts) > 0):
   
          # to debug we draw the ends of the "search line"
          #self.canvas.create_line(math.trunc(pts[0][1]),math.trunc(pts[0][0]),
	  #			  math.trunc(pts[len(pts)-1][1]),
	  #			  math.trunc(pts[len(pts)-1][0]),
	  #		          fill="red")
   
          # go through the "search line" collecting pixel values
          pts_greyvals=np.array([self.pixValBi(pt,img) for pt in pts])
	  # print "pts_greyvals=",pts_greyvals
   
          #if there is a black line on the "search line"
          if np.min(pts_greyvals)<self.BLACK_THRESHOLD:
              ret_val=True
              # compute the centroid along the line
              centroid = np.dot(weights,255-pts_greyvals) / \
			 np.sum(255-pts_greyvals)
       
       return ret_val,centroid,delta

