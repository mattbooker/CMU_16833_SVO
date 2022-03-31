#include "svo/DepthFilter.hpp"
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <stdio.h>

Filter::Filter(float depth_mean, float min_depth, float max_depth) :
    mean(1./depth_mean),
    variance((1./min_depth * 1./min_depth) / 36.),
    a(10),
    b(10),
    min_depth(1./min_depth),
    max_depth(1./max_depth)
{}


void DepthFilter::updateFilter(float estimated_depth, float tau_sq, Filter& filter) {
    boost::math::normal_distribution<float> normal_dist(filter.mean, sqrt(filter.variance + tau_sq));

    float s_sq = 1./((1. / filter.variance) + (1./tau_sq));
    float m = s_sq * ((filter.mean / filter.variance) + (estimated_depth/tau_sq));
    float C1 = (filter.a / (filter.a + filter.b)) * boost::math::pdf(normal_dist, estimated_depth);
    float C2 = (filter.b / (filter.a + filter.b)) * 1/(filter.min_depth);

    float new_mean = C1 * m + C2 * filter.mean;
    float new_variance = C1 * (s_sq + m * m) + C2 * (filter.variance + filter.mean * filter.mean) - new_mean * new_mean;

    float eq_25 = C1 * (filter.a + 1.)/(filter.a + filter.b + 1.) + C2* (filter.a)/(filter.a + filter.b + 1.);
    float eq_26 = C1 * (filter.a + 1.) * (filter.a + 2.) / ((filter.a + filter.b + 1.) * (filter.a + filter.b + 2.)) 
        + C2 * filter.a * (filter.a + 1.) / ((filter.a + filter.b + 1.) * (filter.a + filter.b + 2.));

    
    // Update the filter
    filter.mean = new_mean;
    filter.variance = new_variance;
    filter.a = (eq_26-eq_25)/(eq_25-eq_26/eq_25);
    filter.b = filter.a*(1. - eq_25)/eq_25;
}