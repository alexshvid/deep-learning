//
// Created by SHVID, Alex on 2/20/19.
//

#include "activation.h"


af::array nn::activation_forward(activation a, const af::array &Z)  {
    switch (a) {
        case kSigmoid:
            return af::sigmoid(Z);
        case kRelu:
            return max(Z, 0.0);
        case kLeakyRelu:
            return max(Z, 0.001 * Z);
        case kTanh:
            return af::tanh(Z);
        case kHardTanh:
            return af::clamp(Z, -1, 1);
        default:
            return af::array();

    }
}

af::array nn::activation_backward(activation a, const af::array &dA, const af::array &A)  {
    switch (a) {
        case kSigmoid:
            return dA * (1 - A * A);
        case kRelu:
            return (dA > 0);
        case kLeakyRelu:
            return (dA > 0);
        case kTanh:
            return 1 - af::tanh(A) ^ 2;
        case kHardTanh:
            return 1 - af::clamp(A, -1, 1) ^ 2;
        default:
            return af::array();

    }
}