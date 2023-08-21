## Introduction

<p align='justify'>
Making a self-driving car is a cool system work. We use data from computer vision and sensor fusion to understand the environment around us. We also use data from localization to understand where we are in that environment. The path planning block uses all the above data to decide which manuever to take next. Then it constructs the trajectory for the controller to execute. Here is a short summary of computer vision, deep learning, localization, path planning and control taught in this course. The summary of sensor fusion part is at the following link (https://github.com/Ly2017To/Sensor_Fusion/blob/master/course_summary.md).
</p> 

## Computer Vision and Deep Learning

<p align='justify'>
Computer Vision enables computer to "see" and "understand". Finding lane lines from pictures taken by a camera mounted on a car is an important task for self driving cars. It consists of the following five steps: 1. convert the image to gray scale; 2. apply a Gaussian filter to blur the gray scale image; 3. apply Canny Edge Detector to the blured gray scale image; 4. apply Hough Transform to detect the lines; 5. draw the lines on the image; Applying a Gaussian filter is to reduce the noise of the gray scale image. This image is the input to the Canny Edge Detector, which detects edges based on the gradient intensity of the pixels on the image. The output image of Canny Edge Detector is a binary image with edges, which is the input to Hough Transform to detect lines on it. Hough transform detects lines by transforming edges in the Cartesian coordinate system to curves in the Polar coordinate system and then determing the lines based on the number of intersecting curves. If this number is above a certain threshold, then these edges form a line.
</p> 

<p align='justify'>
In real world applications, additional steps need to be taken. Before using a camera to take pictures, camera calibration is needed to correct images distortions. For pinhole cameras, radial distortion and tangential distortion are the two major kinds of distortions. Images stored in RGB color space might lose some useful information for detecting lanes at some conditions. HSV and HSL are another two color spaces. For edge detections, it increases robustness by combining gradient intensity with thresholding different fields of color spaces. For drawing the lane lines, by applying a perspective transform on the image to have a good view of region and interest and then applying sliding window method to detect lanes based on detected edges.
</p> 

<p align='justify'>
Deep Learning is a fantastic field and is widely applied in computer vision. This course introduces NN(Neural Network), CNN(Convolutional Neural Network) and FCN(Fully Convolutional Network). The following contents provides a brief introduction and it is worthwhile to spend time to read research papers to have a deeper understanding. An NN is comprised of layers of nodes connected with each other, including an input layer, hidden layer(s) and an output layer. Each node is neuron, which has an input, an activation function and an output. The connections of a node with the other nodes are modeled by weights. Thus, from inputs, an output of an NN can be obtained by matrix multiplications of outputs of nodes of each layer and their corresponding weights. Notice that activation functions introduce non-linearily, so an NN can express both linear and non-linear model. The parameters(weights) of a model is learnt by training, which is the process of using a set of data of a particular task to tune these parameters. When the parameters are tuned to satisfy certain criteria, the model can be applied to inference, which is a process to make predictions about the data of the a particular task.
</p>

<p align='justify'>
CNN is widely used in image classification field. A CNN consists of an input layer, convolutional layers, pooling layers, fully connected layers and an output layer. Convolutional layers are used to learn features of input. An analogy is cross correlation used in signal processing to detect pressence of signals. A pooling layer usually follows a convolutional layer to extract the important features. In this way, a CNN learns features of input from lower level to higher level layer by layer. For instance, for an image classification task, lower level features are edges, gradient orientations and so on, while higher level features are more complicated like items, scenes and so on. The last layer of high level features are the input to the fully connected layers to do the classification or regression task. The output of fully connected layer is the input to the output layer to generate the desired output. FCN is mainly used in semantic segmentation field. An FCN consists of an input layer, convolutional layers, pooling layers, deconvolutional layers and an output layer. Comparing with the architecture of CNN, FCN substitudes fully connected layers with deconvolutional layers. Convolutional layers and pooling layers are used to extract features of the input. Deconvolutional layers are used to perform upsampling, which recover the size of extract features to the original input for localizing the features in the original input. This is an important process to achieve pixel-level classification of the input image.
</p>

## Localization

<p align='justify'>
For localization, we apply Bayes Filter, which is a general framework for recursive state estimation. Recursive means only the last training and inference state, current control and current observations are used to estimate the current state. In addition, the motion model describes the prediction step of the filter and the observation model describes the update step of the filter to estimate the new states. Thus, Kalman Filters and Particle Filters are realizations of Bayes Filters. 
</p> 

<p align='justify'>
In Particle Filters, each particle is a single and discrete estimation of the states. A set of particles together comprise a approximate representation of the posterior of the states estimations. The following process describes the working flow of Particle Filters. $X_{t-1}$ is the states of a set of particles at time $t-1$. $u_t$ is the control input at time $t$. $z_t$ is the measurement at time $t$. First, by applying the motion model, the state of each particle in the set $x^{i}_t$ can be obtained from its state at time $t-1$ and $u_t$. Second, by applying the observation model, the weight of each particle $w^{i}_t$ can be updated based on $x^{i}_t$ and $z_t$. This weight measures the corresponding degree of consistencies of a particle with the sensor measurements. Third, resampling makes each particle survive based on its weight to form a new set of particles resampled.  
</p> 

<p align='justify'>
The following explains the math behind Particle Filters. $X$ represents the states. $X^{\prime}$ represents the states of the next time step. $Z$ represents the measurements. $P(X)$ is the prior, which means the distribution of a set of particles. $P(Z|X)$ is the likelihood, which means importance weights. $P(X|Z)$ is the posterior. As for Particle Filters, measurement updates is a resampling process. $P(X^{\prime})$ is the distribution of the above set of particles the next time step and it is the convolution of the transition probability $P(X^{\prime}|X)$ times the prior $P(X)$. The mathematical descriptions of measurement updates and motion updates are shown below.
</p> 

**Measurement Updates**

$P(X|Z) \propto P(Z|X) \cdot P(X)$

**Motion Updates**

$P(X^{\prime}) = \sum P(X^{\prime}|X) \cdot P(X)$

<p align='justify'>
The following table lists the similarities and differences of Histogram Filters, Kalman Filters and Particle Filters in various aspects. 
</p>

|                    | State Space   | Belief        | Efficiency    | In Robotics   |
| -------------------|:-------------:|:-------------:|:-------------:|:-------------:|
| Histogram Filters  | Discrete      | multimodal    | Exponential   | approximate   |
| Kalman Filters     | Continuous    | unimodal      | quadratic     | approximate   | 
| Particle Filters   | Continuous    | multimodal    | not sure yet  | approximate   | 


## Path Plannning

<p align='justify'>
Path Planning includes behaviour planning, prediction and trajectory planning. Behaviour Planning provides guidance to trajectory planners about feasible, safe, legal and efficient manuevers they should plan trajectories for. The inputs to the behaviour planning module are from localization module and prediction module. The output of the behaviour planning module goes into trajectory planning module, which also takes inputs from prediction module and localization module. The output of trajectory planning module goes into motion control. Note that behaviour planning has no responsibility for execution details and collision avoidance. A way to solve behaviour planning is the Finite State Machine, which is a mathematical model to describe system states and states transitions triggered by inputs. For our high way driving example, there are seven states including Ready, Lane Keep, Lane Change Left, Prepare for lane change left, Lane Change Right and Prepare for lane change right. States transitions triggered by inputs come along with costs, which can be modeled by cost functions mathematically. The input to cost functions are predictions, map, speed limit, localization and current state. When designing cost functions, we use different weights to address various problems encountered. The priorities of issues addressed are feasibility, safety, legality, comfort and efficiency. Prediction module takes inputs from localization and sensor fusion and generates predictions of future states of the other moving objects as outputs, which are represented by trajectories and corresponding probabilities. There are two main prediction techniques called modal-based approach and data-driven approach. Modal-based approach uses mathematical modals of motion to predict trajectories, while data-driven approach relies on data and machine learning to learn from.
</p>

<p align='justify'>
A feasible motion planning means to find a sequence of movements in configuration space that moves our robot from start configuration to goal configuration without hitting any obstacles. The properties of motion planning algorithms are completeness and optimality. The completeness means if a solution exists, planner always finds a solution. if no solution exists, terminates and reports failure. The optimality means given a cost function for evaluating a sequence of actions, planner always returns a feasible sequence of actions with minimal costs. There are several classes of motion planning algorithms, including combinatorial methods, potential field, optimal control and sampling methods. Sampling methods use a collision detection module to probe the free space to see if a configuration is in collision or not and the explored parts are stored in a graph structure that can be searched with graph search algorithm. A star algorithm belongs to discrete sampling methods and recursively traverses accessible positions from starting position to goal position guided by a heuristic cost function, so it always finds an optimal path from starting position to goal position. Due to its discrete nature, it may find paths that are not drivable. Hybrid A star algorithm belongs to continuous sampling methods and generates a drivable path by traversing conform to kinematics. It is worthwhile to take time to read research paper about it. The following two tables compare the above two algorithms in various aspects.
</p>

|                                                                        | A Star   | Hybrid A Star | 
| -----------------------------------------------------------------------|:--------:|:-------------:|
| It is a continuous method                                              | False    | True          |
| It uses an optimistic heuristic function to guide grid cell expansion  | True     | True          | 
| It always finds a solution if it exists                                | True     | False         | 
| Solutions it finds are drivable                                        | False    | True          |
| Solutions it finds are always optional                                 | True     | False         |

<p align='justify'>
For generating trajectory of a self driving car on the road, we consider Frenet coordinate system. Frenet has two dimensions $s$ and $d$, among which $s$ represents longitudinal motion and $d$ represents lateral motion. When planning a trajectory in a dynamic environment, it is important to not just compute a sequence of configurations, but to decide when we are going to be in each configuration. Our goal is to generate continuous trajectory. We also have longitudinal constraints and lateral constraints. In addition, we need position continuity and velocity continuity. People feel discomfort when jerk is high. Notice that jerk is the third derivative of position. 
</p>

<p align='justify'>
Let us first consider one dimensional trajectory $s(t)$. Suppose the goal is to minimize the square of jerk in a time interval from $0$ to $t_f$, the trajectory is a fifth order polynomial $s(t) = a_0 + a_1 \cdot t + a_2 \cdot t^2 + a_3 \cdot t^3 + a_4 \cdot t^4 + a_5 \cdot t^5$. Thus, we have six parameters to solve and we can solve it by providing six boundary conditions, such as position of start state, velocity of start state, acceleration of start state, position of goal state, velocity goal state and acceleration of goal state. By taking the above six boundary conditions and the time duration to a polynomial solver, the above six parameters can be generated. The trajectory on the other dimension $d(t)$ can be generated by the same way. There are some points need to be checked for a trajectory solution, including maximum velocity, minimum velocity, maximum acceleration, minimum acceleration and steering angle. When selecting trajectories, distance to obstacles, distance to center line and time to goal need to be considered. In reality, self driving cars have several trajectory planners to use depending on the situations. For example, hybrid A star for parking lots and polynomial trajectory generation for low traffic high way. The following table compares the above driving environmental conditions.
</p>

|                | Unstructed           | Structed                                          | 
| ---------------|:--------------------:|:-------------------------------------------------:|
| Example        | Parking lot, Maze    | High Way, Street Driving                          |
| Rules          | Less Specific Rules  | Predefined Rules                                  | 
| Reference Path | Not Obvious          | Road Structure can be used as a reference         |


## Control

<p align='justify'>
We need to send control commends to drive the car in the real world. For example, throttle, brake, steering and so on. PID (Proportional Integral Derivative) controller is an important algorithm in this field. P stands for Proportional, which compenstates the error you would like to track by multiplying this error with a coefficient $K_p$. If there is only P(proportional) controller, then the output of the controller oscilates and the error oscilates too. So the rate of error needs to be compensated too. Here comes the D(derivative) controller, which compensates it by multiplying the rate of error with a coefficient $K_d$. The rate of error can be calculated as the difference of the current measured error and the last measured error. In general, a PD controller with tuned coefficients may satisfy the requirements. However, we may encounter system bias in the real world. Thus, the I(integral) controller is instroduced to compensate the system bias by multiplying the integral of error for a time interval with a coefficient $K_i$. The output of the PID controller is shown as the formula below.

$u(t) = K_p * e(t) + K_d * \frac{de(t)}{dt} + K_i * \int e(t)dt$

</p>

<p align='justify'>
There are various methods available to tune the coefficients of the above formula. One is SGD (Stochastic Gradient Descent), which I have applied in the course project to optimize the squre of error and tune each coefficient seperately (https://github.com/Ly2017To/Self_Driving_Car/blob/master/PID_Control/writeup.md). Another one is Twiddle that is introduced in the class, which is a local hill climber algorithm to optimize a set of parameters. The peseudo code of Twiddle algorithm is shown below.
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

<p align='justify'>
Model Predictive Control reframes the task of following a trajectory to a optimization problem. The solution to the problem is the optimal trajectory, which is selected based on the minimal costs of multiple trajectories generated by simulating multiple actuator inputs. The optimal trajectory is re-calculated after the first set of actuation commands because our model is only an approximation to the real world. In this way, the actuation commands are optimized in each time step to minimize the cost of our prediceted trajectory. 
</p> 