//
// Created by SHVID, Alex on 2/19/19.
//


#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "deepnet.h"

using namespace af;

af::array load(std::istream& in) {

    int n;
    in >> n;

    float* list = new float[n];

    std::cout << "n=" << n << std::endl;

    for (int i = 0; i <= n; ++i) {
        in >> list[i];
    }

    af::array A = af::array(1, n, list);

    delete [] list;

    return A;
}


int main(int argc, char *argv[])
{

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);


    try {


        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        std::ifstream fileX0("/Users/ashvid/CLionProjects/deep-learning/X0.txt");
        af::array X0 = load(fileX0);

        std::ifstream fileX1("/Users/ashvid/CLionProjects/deep-learning/X1.txt");
        af::array X1 = load(fileX1);

        std::ifstream fileY("/Users/ashvid/CLionProjects/deep-learning/Y.txt");
        af::array Y = load(fileY);

        af::array X =af::join(0, X0, X1);

        /*
        std::cout << "X0=" << std::endl;
        af_print(X0);

        std::cout << "X1=" << std::endl;
        af_print(X1);

        std::cout << "X=" << std::endl;
        af_print(X);

        std::cout << "Y=" << std::endl;
        af_print(Y);
         */

        int n = X.dims(0);
        int m = X.dims(1);

        std::cout << "m = " << m << std::endl;
        std::cout << "n = " << n << std::endl;

        nn::deepnet net(n);

        net.add_layer(40, nn::kTanh);
        net.add_layer(1, nn::kSigmoid);

        net.print();

        net.train(X, Y, 100000, 1.2);

        net.print();

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
