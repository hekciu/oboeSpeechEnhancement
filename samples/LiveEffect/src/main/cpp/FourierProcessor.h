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
    const T PI = 3.14159265358979323846264338327950288419716939937510;
    int samplesNumber = SAMPLES_TO_MODEL;
    int fftInputSize;
    int cooleyTukeyStoragesNum;

    T dftCoeffsRealPart[SAMPLES_TO_MODEL];
    T dftCoeffsImagPart[SAMPLES_TO_MODEL];

    T dftFactorsRealPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];
    T dftFactorsImagPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];

    T idftCoeffsRealPart[SAMPLES_TO_MODEL];
    T idftCoeffsImagPart[SAMPLES_TO_MODEL];

    T idftFactorsRealPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];
    T idftFactorsImagPart[SAMPLES_TO_MODEL][SAMPLES_TO_MODEL];

    T ** storagesFirstPartReal;
    T ** storagesFirstPartImag;
    T ** storagesSecondPartReal;
    T ** storagesSecondPartImag;

    T * currentEvenOutput[2];
    T * currentOddOutput[2];

    T ** inputStoragesFirstPart;
    T ** inputStoragesSecondPart;

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

    void createCooleyTukeyStorages() {
        this->storagesFirstPartReal = new T * [this->cooleyTukeyStoragesNum];
        this->storagesFirstPartImag = new T * [this->cooleyTukeyStoragesNum];
        this->storagesSecondPartReal = new T * [this->cooleyTukeyStoragesNum];
        this->storagesSecondPartImag = new T * [this->cooleyTukeyStoragesNum];

        this->inputStoragesFirstPart = new T * [this->cooleyTukeyStoragesNum];
        this->inputStoragesSecondPart = new T * [this->cooleyTukeyStoragesNum];
        int step = 1;

        for (int i = 0; i < this->cooleyTukeyStoragesNum; i++) {
            step *= 2;
            this->storagesFirstPartReal[i] = new T [this->fftInputSize / step];
            this->storagesFirstPartImag[i] = new T [this->fftInputSize / step];
            this->storagesSecondPartReal[i] = new T [this->fftInputSize / step];
            this->storagesSecondPartImag[i] = new T [this->fftInputSize / step];

            this->inputStoragesFirstPart[i] = new T [this->fftInputSize / step];
            this->inputStoragesSecondPart[i] = new T [this->fftInputSize / step];
        }
    }

    void destroyCooleyTukeyStorages() {
        for (int i = 0; i < this->cooleyTukeyStoragesNum; i++) {
            delete this->storagesFirstPartReal[i];
            delete this->storagesFirstPartImag[i];
            delete this->storagesSecondPartReal[i];
            delete this->storagesSecondPartImag[i];

            delete this->inputStoragesFirstPart[i];
            delete this->inputStoragesSecondPart[i];
        }
        delete this->storagesFirstPartReal;
        delete this->storagesFirstPartImag;
        delete this->storagesSecondPartReal;
        delete this->storagesSecondPartImag;

        delete this->inputStoragesFirstPart;
        delete this-> inputStoragesSecondPart;
    }

    void getFftInputSize() { // need to be called before createStorageForCooleyTukey
        this->fftInputSize = 2;
        this->cooleyTukeyStoragesNum = 1;

        while (this->fftInputSize < this->samplesNumber) {
            this->fftInputSize *= 2;
            this->cooleyTukeyStoragesNum++;
        }
    }
public:
    FourierProcessor() {
        this->getFftInputSize();
        this->createCooleyTukeyStorages();
        this->calculateDftCoeffs();
        this->calculateDftFactors();
        this->calculateIdftCoeffs();
        this->calculateIdftFactors();
    }

    ~FourierProcessor() {
        this->destroyCooleyTukeyStorages();
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

    void fftWithSamplesAddition(T * input, T ** output) {
        T * biggerInput = new T [this->fftInputSize];
        for (int n = 0; n < this->fftInputSize; n++) {
            if (n < this->samplesNumber) {
                biggerInput[n] = input[n];
            } else {
                biggerInput[n] = 0;
            }
        }

        T * biggerOutputReal = new T [this->fftInputSize];
        T * biggerOutputImag = new T [this->fftInputSize];

        T ** biggerOutput = new T * [2];
        biggerOutput[0] = biggerOutputReal;
        biggerOutput[1] = biggerOutputImag;

        this->fft(biggerInput, biggerOutput, this->fftInputSize);

        for (int n = 0; n < this->samplesNumber; n++) {
            output[0][n] = biggerOutput[0][n];
            output[1][n] = biggerOutput[1][n];
        }

        delete[] biggerOutputImag;
        delete[] biggerOutputReal;
        delete[] biggerInput;
        delete[] biggerOutput;
    }

    void fft(T * input, T ** output, int N = -1) {
        N = N == -1 ? this->fftInputSize : N;
        int currentStorageNumber = this->cooleyTukeyStoragesNum - log2(N); // TODO check if valid with int

        if (N == 1) {
            output[0][0] = input[0];
            output[1][0] = 0;
            return;
        }

        for (int i = 0; i < N; i += 2) {
            this->inputStoragesFirstPart[currentStorageNumber][i / 2] = input[i];
            this->inputStoragesSecondPart[currentStorageNumber][i / 2] = input[i + 1];
        }

        this->currentEvenOutput[0] = this->storagesFirstPartReal[currentStorageNumber];
        this->currentEvenOutput[1] = this->storagesFirstPartImag[currentStorageNumber];
        this->currentOddOutput[0] = this->storagesSecondPartReal[currentStorageNumber];
        this->currentOddOutput[1] = this->storagesSecondPartReal[currentStorageNumber];

        this->fft(this->inputStoragesFirstPart[currentStorageNumber], this->currentEvenOutput, N / 2);
        this->fft(this->inputStoragesSecondPart[currentStorageNumber], this->currentOddOutput, N / 2);

        for (int k = 0; k < N / 2; k++) {
            T pReal = this->storagesFirstPartReal[currentStorageNumber][k];
            T pImag = this->storagesFirstPartImag[currentStorageNumber][k];

            T qReal = this->storagesSecondPartReal[currentStorageNumber][k] * cos(-2 * k * this->PI / N);
            T qImag = this->storagesSecondPartImag[currentStorageNumber][k] * sin(-2 * k * this->PI / N);

            output[0][k] = pReal + qReal;
            output[1][k] = pImag + qImag;

            output[0][k + N / 2] = pReal - qReal;
            output[1][k + N / 2] = pImag - qImag;
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

    void ifft(T ** input, T * output) {

    }
};

#endif //SAMPLES_FOURIERPROCESSOR_H
