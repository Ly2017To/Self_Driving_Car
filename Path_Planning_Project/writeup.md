# Path-Planning Project

The goal of this project is to build a path planner that creates smooth, safe trajectories for the car to follow. Also, the driving behavior of the car should obey the various points listed on the rubic.

Input: car's location and sensor fusion data (estimates the location of all the vehicles on the same side of the road) 

Output: a list of x and y global map coordinates, which makes the trajectory of the car to follow.

I will divide my writeup into two parts, path planning and behaviour planning.

## Path Planning 

To start, I tried to implement the path planning algorithm by JMT method (I have commented it in the code). After trying a lot of times, it fails. I think this method will be my future work. Then I switched to apply the alogrithm using spline provided in the Q and A Section. This algorithm using spline to take 5 points to fit a ploynomial as a trajectory for the ego car. The generated path of each step consists of the points of previously generated trajectory that are not consumed by the simulator and newly generated trajectory. These 5 points are seletected as follows, 2 points are selected based on the last points of the trajectory of the previous step and 3 points are selected evenly spread ahead in the frenet coordinate system. Then the points of the newly generated trajectory are added according to the velocity, distance traveled and the polynomial fitted above. 

## Behaviour Planning

Behaviour Planning is done by following the starting code provided in the Q and A Section. By applying finite state machine, I define the car's driving behaviour in different states. First, if there is no slow car ahead of the ego car, go back to the middle lane or increase the speed. If there is slow car ahead of the ego car, decrease the speed and then check the conditions of adjacent lanes to decide whether to change the lane or stay in the lane. The implementation details can be seen in the code.

## Future Work
Trying to apply what I have learnt in the lesson more into this project and to make it work.
