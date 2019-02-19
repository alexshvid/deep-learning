//
// Created by SHVID, Alex on 2/19/19.
//

#ifndef DEEPNN_NNET_H
#define DEEPNN_NNET_H

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

namespace nn {

    struct layer {

        int n;
        af::array W;
        float b;

        layer(int n_x) : n(n_x), W(af::array()), b(0.0f) {
        }

        layer(int n_h, int n_h_prev) : n(n_h), W(af::randu(n_h, n_h_prev, f32) * 0.01), b(0.0f) {
        }

    };

    struct forward_propagation {

        af::array &A;
        layer &l;
        af::array Z;

        forward_propagation(af::array &A, layer &l) : A(A), l(l) {

            Z = af::matmul(l.W, A) + l.b;

        }

    };

    class deepnet {

    public:

        deepnet(std::vector<int> &layer_dims);


        void print() {
            for (auto i = layers.begin(); i != layers.end(); ++i) {
                af_print(i->W);
            }
        }

    private:

        std::vector<layer> layers;

    };


}

#endif //DEEPNN_NNET_H
