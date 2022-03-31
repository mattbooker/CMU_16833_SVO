#include <gtest/gtest.h>
#include "svo/DepthFilter.hpp"
#include <stdio.h>

// Demonstrate some basic assertions.
TEST(FilterTest, UpdateFilter) {
    float depth = 10;
    float min = 5;
    float max = 15;
    float estimated = 8;
    float tau_sq = 2;

    Filter f = Filter(depth, min, max);
    DepthFilter df;

    df.updateFilter(estimated, tau_sq, f);

    // Hand calcualted values
    float expected_a = 0.046278000674289325;
    float expected_b = 200.2622768380564;
    float expected_mean = 0.9381927813491586;
    float expected_variance = 8.50115463941132;

    EXPECT_EQ(f.a, expected_a);
    EXPECT_EQ(f.b, expected_b);
    EXPECT_EQ(f.mean, expected_mean);
    EXPECT_EQ(f.variance, expected_variance);

    // These shouldnt change
    EXPECT_EQ(f.min_depth, min);
    EXPECT_EQ(f.max_depth, max);
}