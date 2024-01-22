//
// Created by jakub on 12.01.2024.
//

#ifndef SAMPLES_FOURIERPROCESSOR_H
#define SAMPLES_FOURIERPROCESSOR_H

#include <cmath>
#include "constants.h"
#include <vector>
#include <complex>

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


    T ** storagesFirstPartInverse;
    T ** storagesSecondPartInverse;

    T * currentEvenInputInverse[2];
    T * currentOddInputInverse[2];

    T ** inputStoragesFirstPartInverseReal;
    T ** inputStoragesFirstPartInverseImag;
    T ** inputStoragesSecondPartInverseReal;
    T ** inputStoragesSecondPartInverseImag;

    int * rearrangedIndices;

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


        this->storagesFirstPartInverse = new T * [this->cooleyTukeyStoragesNum];
        this->storagesSecondPartInverse = new T * [this->cooleyTukeyStoragesNum];

        this->inputStoragesFirstPartInverseReal = new T * [this->cooleyTukeyStoragesNum];
        this->inputStoragesFirstPartInverseImag = new T * [this->cooleyTukeyStoragesNum];
        this->inputStoragesSecondPartInverseReal = new T * [this->cooleyTukeyStoragesNum];
        this->inputStoragesSecondPartInverseImag = new T * [this->cooleyTukeyStoragesNum];
        int step = 1;

        for (int i = 0; i < this->cooleyTukeyStoragesNum; i++) {
            step *= 2;
            this->storagesFirstPartReal[i] = new T [this->fftInputSize / step];
            this->storagesFirstPartImag[i] = new T [this->fftInputSize / step];
            this->storagesSecondPartReal[i] = new T [this->fftInputSize / step];
            this->storagesSecondPartImag[i] = new T [this->fftInputSize / step];

            this->inputStoragesFirstPart[i] = new T [this->fftInputSize / step];
            this->inputStoragesSecondPart[i] = new T [this->fftInputSize / step];


            this->storagesFirstPartInverse[i] = new T [this->fftInputSize / step];
            this->storagesSecondPartInverse[i] = new T [this->fftInputSize / step];

            this->inputStoragesFirstPartInverseReal[i] = new T [this->fftInputSize / step];
            this->inputStoragesFirstPartInverseImag[i] = new T [this->fftInputSize / step];
            this->inputStoragesSecondPartInverseReal[i] = new T [this->fftInputSize / step];
            this->inputStoragesSecondPartInverseImag[i] = new T [this->fftInputSize / step];
        }
    }

    void destroyCooleyTukeyStorages() {
        for (int i = 0; i < this->cooleyTukeyStoragesNum; i++) {
            delete[] this->storagesFirstPartReal[i];
            delete[] this->storagesFirstPartImag[i];
            delete[] this->storagesSecondPartReal[i];
            delete[] this->storagesSecondPartImag[i];

            delete[] this->inputStoragesFirstPart[i];
            delete[] this->inputStoragesSecondPart[i];


            delete[] this->storagesFirstPartInverse[i];
            delete[] this->storagesSecondPartInverse[i];

            delete[] this->inputStoragesFirstPartInverseReal[i];
            delete[] this->inputStoragesFirstPartInverseImag[i];
            delete[] this->inputStoragesSecondPartInverseReal[i];
            delete[] this->inputStoragesSecondPartInverseImag[i];
        }
        delete[] this->storagesFirstPartReal;
        delete[] this->storagesFirstPartImag;
        delete[] this->storagesSecondPartReal;
        delete[] this->storagesSecondPartImag;

        delete[] this->inputStoragesFirstPart;
        delete[] this->inputStoragesSecondPart;


        delete[] this->storagesFirstPartInverse;
        delete[] this->storagesSecondPartInverse;

        delete[] this->inputStoragesFirstPartInverseReal;
        delete[] this->inputStoragesFirstPartInverseImag;
        delete[] this->inputStoragesSecondPartInverseReal;
        delete[] this->inputStoragesSecondPartInverseImag;
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
//        this->createCooleyTukeyStorages();
//        this->calculateDftCoeffs();
//        this->calculateDftFactors();
//        this->calculateIdftCoeffs();
//        this->calculateIdftFactors();
        this->getRearrangedIndices();
    }

    ~FourierProcessor() {
//        this->destroyCooleyTukeyStorages();
        delete rearrangedIndices;
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

    void _indicesHelp(int * input, int N) {
        if (N == 1) {
            return;
        }

        // divide for evens and odds
        int * evens = new int [N / 2];
        int * odds = new int [N / 2];
        for (int i = 0; i < N; i += 2) {
            evens[i / 2] = input[i];
            odds[i / 2] = input[i + 1];
        }

        _indicesHelp(evens, N / 2);
        _indicesHelp(odds, N / 2);

        for (int i = 0; i < N; i++) {
            if (i < N / 2) {
                input[i] = evens[i];
            } else {
                input[i] = odds[i - N / 2];
            }
        }
    }

    void getRearrangedIndices() {
        this->rearrangedIndices = new int [this->fftInputSize];
        for (int i = 0; i < this->fftInputSize; i++) {
            this->rearrangedIndices[i] = i;
        }

        this->_indicesHelp(this->rearrangedIndices, this->fftInputSize);
    }

    void rearrangeComplex(T ** output) {
        T ** copy = new T * [2];
        copy[0] = new T [this->fftInputSize];
        copy[1] = new T [this->fftInputSize];

        for (int i = 0; i < this->fftInputSize; i++) {
            copy[0][i] = output[0][i];
            copy[1][i] = output[1][i];
        }

        for (int i = 0; i < this->fftInputSize; i++) {
            output[0][i] = copy[0][this->rearrangedIndices[i]];
            output[1][i] = copy[1][this->rearrangedIndices[i]];
        }

        delete[] copy[0];
        delete[] copy[1];
        delete[] copy;
    }

    void rearrange(T * output) {
        T * copy = new T [this->fftInputSize];

        for (int i = 0; i < this->fftInputSize; i++) {
            copy[i] = output[i];
        }

        for (int i = 0; i < this->fftInputSize; i++) {
            output[i] = copy[this->rearrangedIndices[i]];
        }
        delete[] copy;
    }


    std::vector<std::complex<T>> fftVectorsRec(const std::vector<std::complex<T>> &input) {
        int N = input.size();

        if (N == 1) {
            return input;
        }

        std::vector<std::complex<T>> odd, even;

        for (int i = 0; i < N; i+= 2) {
            even.push_back(input[i]);
            odd.push_back(input[i + 1]);
        }

        even = this->fftVectorsRec(even);
        odd = this->fftVectorsRec(odd);

        std::vector<std::complex<T>> result(N);
        std::complex<T> multiplier = std::complex<T>(cos((-2.0 * this->PI) / (T)N), sin((-2.0 * this->PI) / (T)N));

        for (int k = 0; k < N / 2; k++) {
            result[k] = even[k] + pow(multiplier, (T)k) * odd[k];
            result[k + N / 2] = even[k] + pow(multiplier, (T)k + (T)N / 2) * odd[k];
        }

        return result;
    }


    void fftVectors(T * input, T ** output) {
        std::vector<std::complex<T>> inputVec(this->fftInputSize);
        for (int i = 0; i < this->fftInputSize; i++) {
            inputVec[i] = std::complex<T>(input[i], 0);
        }

        std::vector<std::complex<T>> outputVec = this->fftVectorsRec(inputVec);

        for (int i = 0; i < this->fftInputSize; i++) {
            output[0][i] = outputVec[i].real();
            output[1][i] = outputVec[i].imag();
        }
    }


    std::vector<std::complex<T>> ifftVectorsRec(const std::vector<std::complex<T>> &input) {
        int N = input.size();

        if (N == 1) {
            return input;
        }

        std::vector<std::complex<T>> odd, even;

        for (int i = 0; i < N; i+= 2) {
            even.push_back(input[i]);
            odd.push_back(input[i + 1]);
        }

        even = this->fftVectorsRec(even);
        odd = this->fftVectorsRec(odd);

        std::vector<std::complex<T>> result(N);
        std::complex<T> multiplier = std::complex<T>(cos((2.0 * this->PI) / (T)N), sin((2.0 * this->PI) / (T)N));

        for (int k = 0; k < N / 2; k++) {
            result[k] = even[k] + pow(multiplier, (T)k) * odd[k];
            result[k + N / 2] = even[k] + pow(multiplier, (T)k + (T)N / 2) * odd[k];
        }

        return result;
    }


    void ifftVectors(T ** input, T * output) {
        std::vector<std::complex<T>> inputVec(this->fftInputSize);
        for (int i = 0; i < this->fftInputSize; i++) {
            inputVec[i] = std::complex<T>(input[0][i], input[1][i]);
        }

        std::vector<std::complex<T>> outputVec = this->ifftVectorsRec(inputVec);

        for (int i = 0; i < this->fftInputSize; i++) {
            output[i] = outputVec[i].real() / this->fftInputSize;
        }
    }


    void fft(T * input, T ** output, int N = -1) {
        N = N == -1 ? this->fftInputSize : N;
        int currentStorageNumber = this->cooleyTukeyStoragesNum - log2(N);

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
        this->currentOddOutput[1] = this->storagesSecondPartImag[currentStorageNumber];

        this->fft(this->inputStoragesFirstPart[currentStorageNumber], this->currentEvenOutput, N / 2);
        this->fft(this->inputStoragesSecondPart[currentStorageNumber], this->currentOddOutput, N / 2);

        for (int k = 0; k < N / 2; k++) {
            T pReal = this->currentEvenOutput[0][k];
            T pImag = this->currentEvenOutput[1][k];

            T qReal = this->currentOddOutput[0][k] * cos((-2 * (T)k * this->PI) / (T)N) -
                    this->currentOddOutput[1][k] * sin((-2 * (T)k * this->PI) / (T)N);

            T qImag = this->currentOddOutput[1][k] * cos((-2 * (T)k * this->PI) / (T)N) +
                    this->currentOddOutput[0][k] * sin((-2 * (T)k * this->PI) / (T)N);

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

    void ifft(T ** input, T * output, int N = -1) {
        N = N == -1 ? this->fftInputSize : N;
        int currentStorageNumber = this->cooleyTukeyStoragesNum - log2(N); // TODO check if valid with int

        if (N == 1) {
            output[0] = input[0][0] / N;
            return;
        }

        for (int i = 0; i < N; i += 2) {
            this->inputStoragesFirstPartInverseReal[currentStorageNumber][i / 2] = input[0][i];
            this->inputStoragesFirstPartInverseImag[currentStorageNumber][i / 2] = input[1][i];
            this->inputStoragesSecondPartInverseReal[currentStorageNumber][i / 2] = input[0][i + 1];
            this->inputStoragesSecondPartInverseImag[currentStorageNumber][i / 2] = input[1][i + 1];
        }

        this->currentEvenInputInverse[0] = this->inputStoragesFirstPartInverseReal[currentStorageNumber];
        this->currentEvenInputInverse[1] = this->inputStoragesFirstPartInverseImag[currentStorageNumber];
        this->currentOddInputInverse[0] = this->inputStoragesSecondPartInverseReal[currentStorageNumber];
        this->currentOddInputInverse[1] = this->inputStoragesSecondPartInverseImag[currentStorageNumber];

        this->ifft(this->currentEvenInputInverse, this->storagesFirstPartInverse[currentStorageNumber], N / 2);
        this->ifft(this->currentOddInputInverse, this->storagesSecondPartInverse[currentStorageNumber], N / 2);

        for (int k = 0; k < N / 2; k++) {
            T p = this->storagesFirstPartInverse[currentStorageNumber][k];

            T q = this->storagesSecondPartInverse[currentStorageNumber][k] * cos(2 * k * this->PI / N);

            output[k] = (p + q) / N;

            output[k + N / 2] = (p - q) / N;
        }
    }
};

#endif //SAMPLES_FOURIERPROCESSOR_H
