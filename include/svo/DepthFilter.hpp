#ifndef DEPTH_FILTER_H_
#define DEPTH_FILTER_H_

struct Filter {
    float mean;
    float variance;
    float a;
    float b;
    float min_depth;
    float max_depth;
    Filter(float depth_mean, float min_depth, float max_depth);
};

class DepthFilter {

    public:
        void updateFilter(float estimated_depth, float tau_sq, Filter& filter);
};

#endif