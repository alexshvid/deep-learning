//
// Created by SHVID, Alex on 2/19/19.
//

#include "nnet.h"


nn::deepnet::deepnet(std::vector<int>& layer_dims) {

    int L = layer_dims.size();

    layers.push_back( layer(layer_dims[0]) );

    for (int l = 1; l < L; ++l) {
        layers.push_back(layer(layer_dims[l], layer_dims[l-1]));
    }

}

