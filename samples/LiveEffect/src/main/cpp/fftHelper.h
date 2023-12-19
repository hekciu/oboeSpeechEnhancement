//
// Created by jakub on 13.12.2023.
//

#ifndef SAMPLES_FFTHELPER_H
#define SAMPLES_FFTHELPER_H

#include <complex>
#include "constants.h"

// pseudecode from here https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

class FftHelper {
private:
    std::complex<float> _i;
    float _PI = 3.14159265359;

    void _split_real(float * samples, float * odds, float * evens, int output_size) {
        for (int i = 0; i < output_size; i++) {
            int odd_index = 2 * i + 1;
            int even_index = 2 * i;

            odds[i] = samples[odd_index];
            evens[i] = samples[even_index];
        }
    }
public:
    FftHelper() {
        this->_i = std::complex<float>(0, 1);

    }

    ~FftHelper() { }

    void calculate_fft(float * inputs, std::complex<float> * outputs, int N) {
        // we assume that length of inputs == FFT_N

        if (N == 1) {
            outputs[0] = inputs[0];
            return;
        }

        int split_outputs_size = N /2;

        float * odds = new float(split_outputs_size);
        float * evens = new float(split_outputs_size);

        this->_split_real(inputs, odds, evens, split_outputs_size);

        std::complex<float> * odds_output = new std::complex<float>(split_outputs_size);
        std::complex<float> * evens_output = new std::complex<float>(split_outputs_size);

        this->calculate_fft(odds, odds_output, split_outputs_size);
        this->calculate_fft(evens, evens_output, split_outputs_size);

        for (int k = 0; k < split_outputs_size; k++) {
            std::complex<float> p = evens_output[k];
            std::complex<float> q = exp((-2 * this->_PI * this->_i) / std::complex<float>(N, 0));

            outputs[k] = p + q;
            outputs[k + split_outputs_size] = p - q;
        }
    }
};

#endif //SAMPLES_FFTHELPER_H
