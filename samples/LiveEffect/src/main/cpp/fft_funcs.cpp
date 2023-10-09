//
// Created by MonkeyMaster on 11.10.2022.
//
#include "fft_funcs.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <valarray>

// Performs a hann window operation and adds two signals, modifies previous data array, since it won't be usable anymore
//void hann_add(Complex* curData, Complex* prevData, int size)
//{
//    for (int i=0; i < size; ++i)
//    {
//        double window = 0.5   * (1 - cos(2*M_PI*i/(size-1)));
//        prevData[i].real(window*(curData[i].real() + prevData[i].real()));
//    }
//    return;
//}

// Performs a hann window operation and adds two signals, modifies previous data array, since it won't be usable anymore
void hann(float* hannCoeffs, int size) {
    for (int i = 0; i < size; ++i) {
        hannCoeffs[i] = 0.5 * (1 - cos(2 * M_PI * i / size)); // Or: pow(sin(2 * M_PI * i / size), 2)
    }
    return;
}

CValArray data_to_c_valarray(Complex* data, int size)
{
    CValArray data_arr(data, size);
    return data_arr;
}

// Cooley–Tukey FFT (in-place, divide-and-conquer)
void fft(CValArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CValArray even = x[std::slice(0, N/2, 2)];
    CValArray  odd = x[std::slice(1, N/2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
    return;
}

void ifft(CValArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
    return;
}

bool float_comp(float a, float b, float absEpsilon, float relEpsilon)
{
    // Check if the numbers are really close -- needed when comparing numbers near zero.
    double diff{ std::abs(a - b) };
    if (diff <= absEpsilon)
        return true;

    // Otherwise fall back to Knuth's algorithm
    return (diff <= (std::max(std::abs(a), std::abs(b)) * relEpsilon));
}

float sine(float x, int n)
{
    // Taylor series sine wave implementation
    float t = x;
    float sine = x;
    float mult = 0.0;
    int prev_factorial = 1;
    // n - precision, taylor series elements
    for ( int a=1; a<n; ++a)
    {
        mult = -x*x/static_cast<float>(((2*a+1)*(2*a)*prev_factorial)); // nth element, without sign
        prev_factorial = (2*a+1)*(2*a)*prev_factorial; // previous factorial in divisor
        // 2nd: x * -x**2 / 3!*1!, 3rd: -x**3 * -x**2 / 5*4*3!
        t *= mult;
        sine += t;
    }
    return sine;
}