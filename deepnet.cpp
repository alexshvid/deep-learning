//
// Created by SHVID, Alex on 2/19/19.
//

#include <iostream>

#include "deepnet.h"

nn::layer& nn::deepnet::prev() {

    if (layers.size() == 0) {
        return input;
    } else {
        return layers[layers.size()-1];
    }

}

void nn::deepnet::add_layer(int n, activation g) {

    layers.push_back(layer(n, g, prev()));

}

void nn::deepnet::train(af::array& X, af::array& Y, int num_iterations, float learning_rate) {

    for (int i = 0; i < num_iterations; ++i) {

        af::array AL = model(X);

        float costValue = cost(AL, Y);

        if (i % 100 == 0) {
            std::cout << "Cost after iteration " << i << ": " << costValue << std::endl;
        }

        backward(AL, Y);

        update_parameters(learning_rate);

    }

}

af::array nn::deepnet::model(af::array& X) {

    af::array A_prev = X;

    for (auto& layer : layers) {

        layer.forward.Z = af::matmul(layer.W, A_prev) + layer.b;
        layer.forward.A = activation_forward(layer.g, layer.forward.Z);

        A_prev = layer.forward.A;
    }

    return A_prev;
}


float nn::deepnet::cost(af::array& AL, af::array& Y) {

    int m = Y.dims(1);

    af::array logprobs = af::log(AL) * Y + af::log(1-AL) * (1-Y);

    float cost = af::sum<float>(logprobs);

    return cost * - 1 / m;

}

af::array nn::deepnet::linear_backward(af::array& dZ, layer& layer) {

    af::array A_prev = layer.prev.forward.A;

    af::array W = layer.W;
    af::array b = layer.b;

    int m = A_prev.dims(1);

    layer.backward.dW = 1 / m * af::matmul(dZ, A_prev.T());
    layer.backward.db = 1 / m * af::sum<float>(dZ, 1);

    af::array dA_prev = af::matmul(W.T(), dZ);
    layer.prev.backward.dA = dA_prev;

    return dA_prev;
}

void nn::deepnet::backward(af::array& AL, af::array& Y) {

    int m = Y.dims(1);

    Y = af::moddims(Y, AL.dims());

    af::array dA = - (Y / AL) - ((1 - Y) / (1 - AL));

    for (auto i = layers.rbegin(); i != layers.rend(); ++i) {

        af::array dZ = activation_backward(i->g, dA, i->forward.A);

        dA = linear_backward(dZ, *i);

    }


}

void nn::deepnet::update_parameters(float learning_rate) {

    for (auto i = layers.begin(); i != layers.end(); ++i) {

        i->W = i->W - learning_rate * i->backward.dW;
        i->b = i->b - learning_rate * i->backward.db;

    }

}