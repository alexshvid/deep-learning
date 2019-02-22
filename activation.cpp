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
            return Z;
    }
}

af::array nn::activation_backward(activation a, const af::array &dA, const af::array &A)  {
    switch (a) {
        case kSigmoid:
            return dA * A * (1 - A);
        case kRelu:
            return A;
        case kLeakyRelu:
            return A;
        case kTanh:
            return dA * (1 - A * A);
        case kHardTanh:
            return dA * (1 - A * A);
        default:
            return A;
    }
}