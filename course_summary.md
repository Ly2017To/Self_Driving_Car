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
We need to send control commends to drive the car in the real world. For example, throttle, brake, steering and so on. PID (Proportional Integral Difference) controller is an important algorithm in this field. P stands for Proportional, which compenstates the error you would like to track by multiplying this error with a coefficient $K_p$. If there is only P(proportional) controller, then the output of the controller oscilates and the error oscilates too. So the rate of error needs to be compensated too. Here comes the D(difference) controller, which compensates it by multiplying the rate of error with a coefficient $K_d$. The rate of error can be calculated as the difference of the current measured error and the last measured error. In general, a PD controller with tuned coefficients may satisfy the requirements. However, we may encounter system bias in the real world. Thus, the I(integral) controller is instroduced to compensate the system bias by multiplying the integral of error for a time interval with a coefficient $K_i$. The output of the PID controller is shown as the formula below.

$u(t) = K_p * e(t) + K_d * \frac{de(t)}{dt} + K_i * \int e(t)dt$

</p>

<p align='justify'>
There are various methods available to tune the coefficients of the above formula. One is SGD (Stochastic Gradient Descent), which I have applied in the course project to optimize the squre of error and tune each coefficient seperately (https://github.com/Ly2017To/Self_Driving_Car/blob/master/PID_Control/writeup.md). Another one is Twiddle that is introduced in the class, which is a local hill climber algorithm to optimize a set of parameters.The peseudo code of Twiddle algorithm is shown below.
</p> 

    Twiddle:
    	p is the parameter vector initialized as 0 
		dp is the step of tunning parameter initialized as 1
		err = evaluate(p) returns the error of using parameters p

		best_err = evaluate(p)
		while sum(p) < Tolerance:
			for p_i in p:
				p_i += dp_i
				err = evaluate(p)
				if err < best_err:
					best_err = err
					dp_i *= 1.1
				else
					p_i -= 2*dp_i
					err = evaluate(p)
					if err < best_err:
						best_err = err
						dp_i *= 1.1
					else
					    p_i += dp_i
						dp_i *= 0.9