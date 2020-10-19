/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  default_random_engine gen;

  // creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(size_t i=0; i<num_particles; i++){
    Particle p;
    p.id=i;
    p.x=dist_x(gen);
    p.y=dist_y(gen);
    p.theta=dist_theta(gen);
    p.weight=1.0;
     
    particles.push_back(p);
  }

  is_initialized=true;

  return;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  //cout << "prediction" <<endl;

  //variable definitions
  double p_theta, p_x, p_y;
  
  p_theta=0.0; 
  p_x=0.0; 
  p_y=0.0;

  default_random_engine gen;
  
  for(size_t i=0; i<num_particles; i++){

    // make the prediction of x, y and theta based on velocity and yaw rate
    // avoid divide by zero
    if(fabs(yaw_rate)>1e-4){
      p_theta=particles[i].theta+delta_t*yaw_rate;
      p_x=particles[i].x+velocity/yaw_rate*(sin(p_theta)-sin(particles[i].theta));
      p_y=particles[i].y+velocity/yaw_rate*(cos(particles[i].theta)-cos(p_theta)); 
    }else{
      p_theta=particles[i].theta;
      p_x=particles[i].x+velocity*delta_t*cos(p_theta);
      p_y=particles[i].y+velocity*delta_t*sin(p_theta);
    }
    
    // creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(p_x, std_pos[0]);
    normal_distribution<double> dist_y(p_y, std_pos[1]);
    normal_distribution<double> dist_theta(p_theta, std_pos[2]);

    particles[i].x=dist_x(gen);
    particles[i].y=dist_y(gen);
    particles[i].theta=dist_theta(gen);
  }

  //cout << "finished" <<endl;

  return;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

   for(size_t i=0; i<observations.size(); i++){
     double distance_min=numeric_limits<double>::max();
     for(size_t j=0; j<predicted.size(); j++){
	double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
	if (distance<distance_min){
 	  distance_min=distance;
	  observations[i].id=j;
	}
     }
   }

   return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   
   //cout << "updateWeights" <<endl;

   //for normalizing the final weights
   double weights_sum=0;

   //Iterate throught each particle
   for(size_t i=0; i<num_particles; i++){

     double x_part = particles[i].x;  
     double y_part = particles[i].y;
     double theta = particles[i].theta;

     //convert the observations from car coordinates to map coordinates 
     vector<LandmarkObs> particle_observations;   
     for(size_t j=0; j<observations.size(); j++){
       double x_obs = observations[j].x;
       double y_obs = observations[j].y;
       double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
       double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

       LandmarkObs observation;
       observation.x = x_map;
       observation.y = y_map;
       observation.id = -1;
	
       particle_observations.push_back(observation);    
     }

     //calculate predicted_landmarks based on sensor range and particle positions
     vector<LandmarkObs> predicted_landmarks;
     for(size_t j=0; j<map_landmarks.landmark_list.size(); j++){
	LandmarkObs landmark;
	landmark.id = map_landmarks.landmark_list[j].id_i;
	landmark.x = map_landmarks.landmark_list[j].x_f;
	landmark.y = map_landmarks.landmark_list[j].y_f;

	if(dist(landmark.x, landmark.y, x_part, y_part)<sensor_range){
	   predicted_landmarks.push_back(landmark);
	}
     }

     //apply data association function
     dataAssociation(predicted_landmarks,particle_observations);

     //initialize the weight of this particle every time when update this weight
     double particle_weight=1.0;

     //calculate the weight of the particle 
     for(size_t j=0; j<particle_observations.size(); j++){
       //cout<<particle_observations[j].id<<endl;
       particle_weight*=multiv_prob(std_landmark[0], std_landmark[1], particle_observations[j].x, particle_observations[j].y, predicted_landmarks[particle_observations[j].id].x, predicted_landmarks[particle_observations[j].id].y);
     }

     particles[i].weight=particle_weight;
     weights_sum+=particle_weight;
   }

   //normalize the weights
   for(size_t i=0; i<num_particles; i++){
     weights.push_back(particles[i].weight/weights_sum);
   }

   //cout << "finished" << endl;
   

   return;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   //cout << "resample" << endl;

   //store the resampled particles
   vector<Particle> particles_resampled;

   //generate random index based on each particle's weight
   random_device rd;
   mt19937 gen(rd());
   discrete_distribution<int> distribution(weights.begin(),weights.end());
   
   //resample the particles
   for(size_t i=0; i<num_particles; i++){
     int index = distribution(gen);
     particles_resampled.push_back(particles[index]);
     //cout << index <<" "<<weights[i]<< endl;
   }

   weights.clear();
   particles=particles_resampled;

   //cout << "finished" << endl;

   return;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
