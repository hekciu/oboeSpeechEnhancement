//
// Created by jakub on 12.01.2024.
//

#ifndef SAMPLES_FOURIERPROCESSOR_H
#define SAMPLES_FOURIERPROCESSOR_H

#include <cmath>
#include "constants.h"

//const float E = 2.718281828459;

template<typename T = float>
class FourierProcessor {
private:
    const T PI = 3.14159265359;
    int samplesNumber = SAMPLES_TO_MODEL;
    T dftCoeffsRealPart[SAMPLES_TO_MODEL];
    T dftCoeffsImagPart[SAMPLES_TO_MODEL];

    T dftFactorsRealPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];
    T dftFactorsImagPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];

    T idftCoeffsRealPart[SAMPLES_TO_MODEL];
    T idftCoeffsImagPart[SAMPLES_TO_MODEL];

    T idftFactorsRealPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];
    T idftFactorsImagPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];

    void calculateDftCoeffs() {
        for (int n = 0; n < this->samplesNumber; n++) {
            this->dftCoeffsRealPart[n] = cos(-2 * this->PI * n / this->samplesNumber);
            this->dftCoeffsImagPart[n] = sin(-2 * this->PI * n / this->samplesNumber);
        }
    }

    void calculateDftFactors() { // need to be called after 'calculateDftCoeffs'
        for (int n = 0; n < this->samplesNumber; n++) {
            T coeffModulus = sqrt(pow(this->dftCoeffsRealPart[n], 2) +
                                      pow(this->dftCoeffsImagPart[n], 2));

            T coeffAngle = atan2(this->dftCoeffsImagPart[n], this->dftCoeffsRealPart[n]);
            for (int k = 0; k < this->samplesNumber; k++) {
                this->dftFactorsRealPart[n][k] = pow(coeffModulus, k) * cos(coeffAngle * k);
                this->dftFactorsImagPart[n][k] = pow(coeffModulus, k) * sin(coeffAngle * k);
            }
        }
    }

    void calculateIdftCoeffs() {
        for (int n = 0; n < this->samplesNumber; n++) {
            this->idftCoeffsRealPart[n] = cos(2 * this->PI * n / this->samplesNumber);
            this->idftCoeffsImagPart[n] = sin(2 * this->PI * n / this->samplesNumber);
        }
    }

    void calculateIdftFactors() { // need to be called after 'calculateIdftCoeffs'
        for (int n = 0; n < this->samplesNumber; n++) {
            T coeffModulus = sqrt(pow(this->idftCoeffsRealPart[n], 2) +
                                      pow(this->idftCoeffsImagPart[n], 2));

            T coeffAngle = atan2(this->idftCoeffsImagPart[n], this->idftCoeffsRealPart[n]);

            for (int k = 0; k < this->samplesNumber; k++) {
                this->idftFactorsRealPart[n][k] = pow(coeffModulus, k) *
                                                  cos(coeffAngle * k) / this->samplesNumber;
                this->idftFactorsImagPart[n][k] = pow(coeffModulus, k) *
                                                  sin(coeffAngle * k) / this->samplesNumber;
            }
        }

    }
public:
    FourierProcessor() {
        this->calculateDftCoeffs();
        this->calculateDftFactors();
        this->calculateIdftCoeffs();
        this->calculateIdftFactors();
    }

    void dft(T * input, T ** output) {
        for (int k = 0; k < this->samplesNumber; k++) {
            output[0][k] = 0; // real part
            output[1][k] = 0; // imag part

            for (int n = 0; n < this->samplesNumber; n++) {
                output[0][k] += input[n] * this->dftFactorsRealPart[k][n];
                output[1][k] += input[n] * this->dftFactorsImagPart[k][n];
            }
        }
    }

    void idft(T ** input, T * output) {
        for (int k = 0; k < this->samplesNumber; k++) {
            output[k] = 0;

            for (int n = 0; n < this->samplesNumber; n++) {
                output[k] += input[0][n] * this->idftFactorsRealPart[k][n] -
                             input[1][n] * this->idftFactorsImagPart[k][n];
            }
        }
    }
};

#endif //SAMPLES_FOURIERPROCESSOR_H
