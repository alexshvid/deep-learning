//
// Created by SHVID, Alex on 2/20/19.
//

#ifndef DEEPNN_ACTIVATION_H
#define DEEPNN_ACTIVATION_H

#include <arrayfire.h>

namespace nn {

    enum activation {
        kSigmoid = 0,
        kRelu = 1,
        kLeakyRelu = 2,
        kTanh = 3,
        kHardTanh = 4
    };

    af::array activation_forward(activation a, const af::array &Z);

    af::array activation_backward(activation a, const af::array &dA, const af::array &A);

}

#endif //DEEPNN_ACTIVATION_H
