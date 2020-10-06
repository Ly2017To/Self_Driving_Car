# **PID Controller** 

**PID Controller on the vehicle**

The goals / steps of this project are the following:
* Build the PID controller to drive e a vehicle safely on the simulator
* Reflect on your work in a written report

### Reflections

My pipeline consisted of the following 3 steps.
1. Based on the course and the starter code, design the PID controller
2. Applying the SGD method to tune the hyper parameters and test on the simulator
3. Using the final tunned hyper parameters to drive the vehicle on the simulator



The challenging part of this project is to tune the hyper parameters of the PID controller (step 2).  

P(proportional) component measures the vehicle's cte(cross tracking error) and the product with its coefficient Kp reacts directly to this cte. I(integral) component records the vehicle's cte in a certain amount of time and the product with its coefficient Ki corrects vehicle's system bias. D(differential) component computes the difference between the vehicle's current cte and its previous cte. The hyper parameters tunning process is described below.

The SGD used is to minimize is the square of total error as the function SGD in the file PID.cpp. At the beginning, I tried to tune all the three parameters Kp, Ki and Kd together, but it does not work. Then, I tried to use SGD to tune the parameters in the following way. The learning rate of the SGD tuner is set to 0.1 and the stop criteria of the SGD is that the absolute value of cte is within a certain tolerance value(0.05). First, tune the proportional parameter Kp. Set the value of Kp initially as 1.0 and the other two hyper parameters Ki and Kd as 0.0, when the stop criteria reaches, keep this value of Kp. It can be observed that the vehicle drives oscillates wilder and wilder later and then out of the track, which implies that a differential component is needed to correct this behavior. Then keep the value of Kp and tune the differential coefficient Kd by following the same procedure as described before. Finally, keep the value of Kp and Kd, tune Ki in the same way as tuning the previous two parameters.

At the end, the PID controller's parameter is Kp=0.236 and Ki=0.003 and Kd=0.934. By applying this set of hyperparameters, this vehicle is able to drive safely with a speed between 31 and 33 MPH.


The advantage of my current solution is that the process of tunning hyper parameter is easy to implement and fast. One shortcoming of my current implementation is that the vehicle oscillates, sometimes a bit wild, especially when it finishes a turn and enters a straight road. A possible improvement would be to develope a more sophisticated PID controller to take the speed and throttle in to consideration.