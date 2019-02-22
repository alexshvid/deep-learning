//
// Created by SHVID, Alex on 2/19/19.
//

#ifndef DEEPNN_NNET_H
#define DEEPNN_NNET_H

#include "activation.h"

namespace nn {

    struct parameters {

        af::array W;
        float b;

    };

    struct forward_propagation {

        af::array Z;
        af::array A;

    };

    struct backward_propagation {

        af::array dA;
        af::array dW;
        float     db;

    };

    struct layer {

        int i;
        int n;
        activation g;

        layer* prev;

        parameters params;
        forward_propagation  forward;
        backward_propagation backward;

        layer(int n) : i(0), n(n), g(kInput), prev(this) {
            params.W = af::array();
            params.b = 0.0f;
        }

        layer(layer* prev, int n, activation g) : i(prev->i+1), n(n), g(g), prev(prev) {
            params.W = af::randu(n, prev->n, f32) * 0.01;
            params.b = 0.0f;
        }

    };

    class deepnet {

    public:

        deepnet(int x_n) : input(x_n) {}

        virtual ~deepnet();

        void add_layer(int n, activation g);

        void train(af::array& X, af::array& Y, int num_iterations, float learning_rate);

        void print();

    private:

        layer* prev();

        af::array forward_propagate(af::array& X);

        float cost(af::array& AL, af::array& Y);

        void linear_backward(af::array& dZ, layer* layer);

        void backward_propagate(af::array& AL, af::array& Y);

        void update_parameters(float learning_rate);

        layer input;
        std::vector<layer*> layers;

    };


}

#endif //DEEPNN_NNET_H
