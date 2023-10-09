//
// Created by MonkeyMaster on 11.10.2022.
//

#ifndef SAMPLES_FFT_FUNCS_H
#define SAMPLES_FFT_FUNCS_H
#include <valarray>
#include <complex>

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CValArray;

void hann(float* hannCoeffs, int size);
CValArray data_to_c_valarray(Complex* data, int size);
void fft(CValArray& x);
void ifft(CValArray& x);
void complex_abs(CValArray& x); // Perhaps return a float* to a new float array, then use it as neural net input
bool float_comp(float a, float b, float absEpsilon, float relEpsilon);
float sine(float x, int n);

#endif //SAMPLES_FFT_FUNCS_H
