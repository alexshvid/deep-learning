//
// Created by SHVID, Alex on 2/19/19.
//


#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

#include "deepnet.h"

using namespace af;


int main(int argc, char *argv[])
{
    try {


        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        nn::deepnet net(4);

        net.add_layer(3, nn::kTanh);
        net.add_layer(1, nn::kSigmoid);

        net.print();

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
