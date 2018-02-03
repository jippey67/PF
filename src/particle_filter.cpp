/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

  // set up gaussian distribution
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  // set up the particles
	for (unsigned int i=0; i<num_particles; ++i) {
		Particle part;
		part.id = i;
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);
		part.weight = 1.0;
		weights.push_back(1.0);
		particles.push_back(part);
	}

  // now initialization is done
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // set up gaussian distribution
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

  // if yaw rate is large enough use standard formulas for prediction, otherwise assume particle
  // moves in a straight line, to prevent division by (almost) zero problems
	if (fabs(yaw_rate) > 0.001) {
		for (unsigned int i=0; i<num_particles; ++i) {
			Particle part = particles[i];
			part.x += velocity/yaw_rate*(sin(part.theta+yaw_rate*delta_t)-sin(part.theta)) + dist_x(gen);
			part.y += velocity/yaw_rate*(cos(part.theta)-cos(part.theta+yaw_rate*delta_t)) + dist_y(gen);
			part.theta += yaw_rate * delta_t + dist_theta(gen);
			particles[i] = part;
		}
	} else {
		for (unsigned int i=0; i<num_particles; ++i) {
			Particle part = particles[i];
			part.x += velocity * cos(part.theta) * delta_t + dist_x(gen);
			part.y += velocity * sin(part.theta) * delta_t + dist_y(gen);
			part.theta += yaw_rate * delta_t + dist_theta(gen);
			particles[i] = part;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	//   NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  // iterate over all observations and find closest landmark by choosing shortest Euclidian distance
	for (int i = 0; i < observations.size(); i++) {
		double shortest_d = 1e9; // initialize to high value
		unsigned int chosen_id = 0;

		for (int j = 0; j < predicted.size(); j++) {
			double x = observations[i].x - predicted[j].x;
			double y = observations[i].y - predicted[j].y;
			double d = sqrt(x * x + y * y);
			if (d < shortest_d) {
				shortest_d = d;
				chosen_id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  // get map coordinates for particles
  for (int i = 0; i < particles.size(); i++) {
    std::vector<LandmarkObs> observations_map;
    particles[i].weight = 1;
    for (LandmarkObs observs : observations) {
      LandmarkObs obs_t;
      obs_t.id = observs.id;
      obs_t.x = observs.x * cos(particles[i].theta) - observs.y * sin(particles[i].theta) + particles[i].x;
      obs_t.y = observs.x * sin(particles[i].theta) + observs.y * cos(particles[i].theta) + particles[i].y;
      observations_map.push_back(obs_t);

      // calculate distance to each landmark
      vector <double> distances;
      for (Map::single_landmark_s landmks : map_landmarks.landmark_list) {
        double distance = dist(landmks.x_f, landmks.y_f, obs_t.x, obs_t.y);
        distances.push_back(distance);
      }

      // find the id of the closest landmark
      vector<double>::iterator result = min_element(begin(distances), end(distances));
      Map::single_landmark_s lm = map_landmarks.landmark_list[distance(begin(distances), result)];
      obs_t.id = lm.id_i;

      // calculate probability
      double normalizer = 1./(2*M_PI*std_landmark[0]*std_landmark[1]);
      double power = -1 * (pow((obs_t.x - lm.x_f),2)/(2*pow(std_landmark[0],2)) + pow((obs_t.y-lm.y_f),2)/(2*pow(std_landmark[1],2)));
      particles[i].weight *= normalizer * exp(power);
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> particles_tmp;
	std::default_random_engine int_eng{};
	std::uniform_int_distribution<> int_distribution{0, num_particles}; // type of engine
	std::default_random_engine real_eng{};
	std::uniform_real_distribution<> real_distribution{0, 1}; // type of engine

	int index = int_distribution(int_eng);

	double beta = 0.0;

	double nw = 0;

	for (int i = 0; i < particles.size(); i++) {
		if (nw < particles[i].weight)
			nw = particles[i].weight;
	}

	// code borrowed from the classroom lessons
	for (int i = 0; i < num_particles; i++) {
		beta += real_distribution(real_eng) * 2.0 * nw;
		while (beta > particles[index].weight) {
			beta -= particles[index].weight;
			index = (index + 1) % num_particles;
		}
		particles_tmp.push_back(particles[index]);

	}
	// set resampled particles
	particles = particles_tmp;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
