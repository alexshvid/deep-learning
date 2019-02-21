//
// Created by SHVID, Alex on 2/19/19.
//

#ifndef DEEPNN_NNET_H
#define DEEPNN_NNET_H

#include "activation.h"

namespace nn {

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

        int n;
        af::array W;
        float b;
        activation g;

        layer& prev;

        forward_propagation  forward;
        backward_propagation backward;

        layer(int n) : n(n), W(af::array()), b(0.0f), g(kRelu), prev(*this) {
        }

        layer(int n, activation g, layer& prev) : n(n), W(af::randu(n, prev.n, f32) * 0.01), b(0.0f), g(g), prev(prev) {
        }

    };

    class deepnet {

    public:

        deepnet(int x_n) : input(x_n) {}

        void add_layer(int n, activation g);

        void train(af::array& X, af::array& Y, int num_iterations, float learning_rate);

        void print() {
            for (auto &i : layers) {
                af_print(i.W);
            }
        }

    private:

        layer& prev();

        af::array model(af::array& X);

        float cost(af::array& AL, af::array& Y);

        af::array linear_backward(af::array& dZ, layer& layer);

        void backward(af::array& AL, af::array& Y);

        void update_parameters(float learning_rate);

        layer input;
        std::vector<layer> layers;

    };


}

#endif //DEEPNN_NNET_H
