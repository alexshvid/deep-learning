//
// Created by SHVID, Alex on 2/19/19.
//

#include <iostream>

#include "deepnet.h"

nn::deepnet::~deepnet() {
    for (auto i : layers) {
        delete i;
    }
}

void nn::deepnet::print() {
    std::cout << "layer[" << input.i << "]=" << input.n << std::endl;
    for (auto &i : layers) {
        std::cout << "layer[" << i->i << "]=" << i->n << ", g=" << i->g << ", prev=" << i->prev->i << std::endl;
    }
}

nn::layer* nn::deepnet::prev() {

    if (layers.size() == 0) {
        return &input;
    } else {
        return layers[layers.size()-1];
    }

}

void nn::deepnet::add_layer(int n, activation g) {
    layers.push_back(new layer(prev(), n, g));
}

void nn::deepnet::train(af::array& X, af::array& Y, int num_iterations, float learning_rate) {

    for (int i = 0; i < num_iterations; ++i) {

        af::array AL = forward_propagate(X);

        float costValue = cost(AL, Y);

        if (i % 1000 == 0) {
            std::cout << "Cost after iteration " << i << ": " << costValue << std::endl;
        }

        backward_propagate(AL, Y);

        update_parameters(learning_rate);

    }

}

af::array nn::deepnet::forward_propagate(af::array& X) {

    input.forward.A = X;

    af::array A_prev = X;

    for (auto layer : layers) {

        layer->forward.Z = af::matmul(layer->params.W, A_prev) + layer->params.b;
        layer->forward.A = activation_forward(layer->g, layer->forward.Z);

        A_prev = layer->forward.A;
    }

    return A_prev;
}


float nn::deepnet::cost(af::array& AL, af::array& Y) {

    int m = Y.dims(1);

    af::array logprobs = af::log(AL) * Y + af::log(1-AL) * (1-Y);

    float cost = af::sum<float>(logprobs);

    return cost * - 1 / m;

}

void nn::deepnet::linear_backward(af::array& dZ, layer* layer) {

    af::array A_prev = layer->prev->forward.A;

    af::array W = layer->params.W;
    af::array b = layer->params.b;

    int m = A_prev.dims(1);

    layer->backward.dW = af::matmul(dZ, A_prev.T()) / m;
    layer->backward.db = af::sum<float>(dZ) / m;

    if (layer->i > 1) {
        layer->prev->backward.dA = af::matmul(W.T(), dZ);
    }

}

void nn::deepnet::backward_propagate(af::array& AL, af::array& Y) {

    int m = Y.dims(1);

    Y = af::moddims(Y, AL.dims());

    //af::array dAL = - (Y / AL) - (1 - Y) / (1 - AL);

    af::array dAL = AL - Y;

    auto i = layers.rbegin();
    (*i)->backward.dA = dAL;

    for (; i != layers.rend(); ++i) {

        layer *layer = *i;

        af::array dZ = activation_backward(layer->g, layer->backward.dA, layer->forward.A);

        linear_backward(dZ, layer);

    }


}

void nn::deepnet::update_parameters(float learning_rate) {

    for (auto i : layers) {

        i->params.W = i->params.W - learning_rate * i->backward.dW;
        i->params.b = i->params.b - learning_rate * i->backward.db;

    }

}