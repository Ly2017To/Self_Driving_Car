## Introduction

<p align='justify'>
Making a self-driving car is a cool system work. We use data from computer vision and sensor fusion to understand the environment around us. We also use data from localization to understand where we are in that environment. The path planning block uses all the above data to decide which manuever to take next. Then it constructs the trajectory for the controller to execute. Here is a short summary of computer vision, deep learning, localization, path planning and control taught in this course. The summary of sensor fusion part is at the following link (https://github.com/Ly2017To/Sensor_Fusion/blob/master/course_summary.md).
</p> 

## Computer Vision

<p align='justify'>

</p> 

## Deep Learning

<p align='justify'>

</p> 

## Localization

<p align='justify'>

</p> 

## Path Plannning

<p align='justify'>

</p> 

## Control

<p align='justify'>
We need to send control commends to drive the car in the real world. For example, throttle, brake, steering and so on. PID (Proportional Integral Difference) controller is an important algorithm in this field. P stands for Proportional, which compenstates the error you would like to track by multiplying this error with a coefficient. If there is only P(proportional) controller, then the output of the controller oscilates and the error oscilates too. So the rate of error needs to be compensated too. Here comes the D(difference) controller, which compensates it by multiplying the rate of error with a coefficient. The rate of error can be calculated as the difference of the current measured error and the last measured error. In general, a PD controller with tuned coefficients may satisfy the requirements. However, we may encounter system bias in the real world. Thus, the I(integral) controller is instroduced to compensate the system bias by multiplying the integral of error for a time interval with a coefficient. 



</p>