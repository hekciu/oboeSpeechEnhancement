/*
 * Copyright 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAMPLES_FULLDUPLEXPASS_H
#define SAMPLES_FULLDUPLEXPASS_H

#include "FullDuplexStream.h"
#include "fft_funcs.h"
#include <math.h>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>

#include "../jni/c_api.h"
#include "../jni/common.h"
#include "../jni/builtin_ops.h"
#include <android/log.h>

#define  LOG_TAG    "testjni"
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

class FullDuplexPass : public FullDuplexStream {
public:
    // Flags
    bool NET_PASS = true;       // Pass the input through the neural network
    bool DB_BOOST = false;      // Needed, because passing samples through neural network makes them quieter.
                                    // Turn off, if testing on loudspeakers
    bool FOURIER_TFS = true;    // Process the input through FFT and iFFT
    bool TEST_SINE = false;     // Using sine wave as an input
    bool TEST_BUFFER = false;    // Verifying the values of FFT frame with exactly the same input block.

    // Params
    bool hannCoeffsCalculated;
    static constexpr int frame_len = 128;
    bool firstCallback;
    int prevModelType;
    int modelDebugType;
    int callbackNum;
    float callbackTimeSum;
    int part_sine_num;
    float f_samp;
    float f_sig;

    // Arrays
    float* RNNInput;
    float* hiddenLayerInputs;
    float* hiddenLayerOutputs;
    float* hannCoeffs;
    float* prevInputFloats1;
    float* prevInputFloats2;
    float* prevInputFloats3;
    float* sinesig;
    float* outputBuffer;
    float* outputBufferDebug;
    float* loadedBuffer;
    Complex* complex_sig;
    CValArray c_valarray_sig;

    // Output file stream
    std::ofstream fftData;

    // TFLite
    TfLiteModel * model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter * interpreter;
    TfLiteTensor * input_tensor;
    TfLiteTensor * hp1_h;
    TfLiteTensor * hp1_c;
    TfLiteTensor * hp2_h;
    TfLiteTensor * hp2_c;
    TfLiteTensor * hp3_h;
    TfLiteTensor * hp3_c;
    const TfLiteTensor * hl1_h;
    const TfLiteTensor * hl1_c;
    const TfLiteTensor * hl2_h;
    const TfLiteTensor * hl2_c;
    const TfLiteTensor * hl3_h;
    const TfLiteTensor * hl3_c;

    const TfLiteTensor * output_tensor;
    FullDuplexPass() { // Instance created and constructor called when app starts

        hannCoeffsCalculated = false;
        hannCoeffs = new float[frame_len]{};
        prevInputFloats1 = new float[64]{};
        prevInputFloats2 = new float[64]{};
        prevInputFloats3 = new float[64]{};
        outputBuffer = new float[64]{};
        outputBufferDebug = new float[64]{};
        sinesig = new float[256];
        complex_sig = new Complex[frame_len];
        hannCoeffsCalculated = false;
        firstCallback = true;
        RNNInput = new float[128]{};
        hiddenLayerInputs = new float[6*128]{}; // 0 initialize hidden states
        hiddenLayerOutputs = new float[6*128]{};
        part_sine_num = 0;

        callbackNum = 0;
        callbackTimeSum = 0.0;
        f_samp = 8000.0;
        f_sig = 440.0;

        loadedBuffer = new float[128]{};
        if(TEST_BUFFER)
        {
            std::ifstream bufferData("/data/data/com.google.oboe.samples.liveeffect/buffer.txt");

            std::string line;
            int it = 0;
            while (std::getline(bufferData, line))
            {
                std::istringstream iss(line);
                float a;
                if (!(iss >> a)) { break; } // error
                loadedBuffer[it] = a;
                ++it;
            }
        }

//        model = TfLiteModelCreateFromFile("/data/data/com.google.oboe.samples.liveeffect/model_lstm_stateful.tflite");
//        options = TfLiteInterpreterOptionsCreate();
//        interpreter = TfLiteInterpreterCreate(model, options);
//        TfLiteInterpreterAllocateTensors(interpreter);
//        input_tensor =
//                TfLiteInterpreterGetInputTensor(interpreter, 3);
//        hp1_h = TfLiteInterpreterGetInputTensor(interpreter, 6);
//        hp1_c = TfLiteInterpreterGetInputTensor(interpreter, 1);
//        hp2_h = TfLiteInterpreterGetInputTensor(interpreter, 2);
//        hp2_c = TfLiteInterpreterGetInputTensor(interpreter, 5);
//        hp3_h = TfLiteInterpreterGetInputTensor(interpreter, 0);
//        hp3_c = TfLiteInterpreterGetInputTensor(interpreter, 4);
//
//        output_tensor =
//                TfLiteInterpreterGetOutputTensor(interpreter, 1);
//        hl1_h = TfLiteInterpreterGetOutputTensor(interpreter, 5);
//        hl1_c = TfLiteInterpreterGetOutputTensor(interpreter, 4);
//        hl2_h = TfLiteInterpreterGetOutputTensor(interpreter, 0);
//        hl2_c = TfLiteInterpreterGetOutputTensor(interpreter, 3);
//        hl3_h = TfLiteInterpreterGetOutputTensor(interpreter, 6);
//        hl3_c = TfLiteInterpreterGetOutputTensor(interpreter, 2);


//        model = TfLiteModelCreateFromFile("/data/data/com.google.oboe.samples.liveeffect/model_lstm.tflite");
//        options = TfLiteInterpreterOptionsCreate();
//        interpreter = TfLiteInterpreterCreate(model, options);
//        TfLiteInterpreterAllocateTensors(interpreter);
//        input_tensor =
//                TfLiteInterpreterGetInputTensor(interpreter, 0);
//        output_tensor =
//                TfLiteInterpreterGetOutputTensor(interpreter, 0);

        modelDebugType = 0;
        prevModelType = -1; // -1: Model not loaded
        ALOG("FullDuplexPass() called!!");

    }

    ~FullDuplexPass() { // Destructor called when app shuts down
        delete[] prevInputFloats1;
        delete[] prevInputFloats2;
        delete[] prevInputFloats3;
        delete[] sinesig;
        delete[] complex_sig;
        delete[] hannCoeffs;
        delete[] RNNInput;
        delete[] outputBuffer;
        delete[] outputBufferDebug;
        delete[] hiddenLayerOutputs;
        delete[] hiddenLayerInputs;
    }

    virtual oboe::DataCallbackResult
    onBothStreamsReady(
            std::shared_ptr<oboe::AudioStream> inputStream,
            const void *inputData,
            int   numInputFrames,
            std::shared_ptr<oboe::AudioStream> outputStream,
            void *outputData,
            int   numOutputFrames)
    {
        // Copy the input samples to the output with a little arbitrary gain change.
        ALOG("Signal processing type: %d", inputStream->getSignalProcessing()); // 0: NN, 1: FFT, 2: Copy
        ALOG("FullDuplexPass::onBothStreamsReady() called!");
        ALOG("Model type: %d", inputStream->getModelType()); // 0: LSTM, 1: GRU

        // Set flags
        switch(inputStream->getSignalProcessing()) {
            case 0:
                NET_PASS = true;
                FOURIER_TFS = true;
                break;
            case 1:
                NET_PASS = false;
                FOURIER_TFS = true;
                break;
            case 2:
                NET_PASS = false;
                FOURIER_TFS = false;
                break;
        }

        ALOG("net_pass: %d, fourier_tfs: %d", NET_PASS, FOURIER_TFS);

        if (NET_PASS && (inputStream->getModelType() != prevModelType)) {

            // Critical Error/Memory Leak || t0nik 03.01.2023
//            // Delete previous TFLite model parameters
//            TfLiteInterpreterDelete(interpreter);
//            TfLiteInterpreterOptionsDelete(options);
//            TfLiteModelDelete(model);

            switch (inputStream->getModelType()) {
                case 0:
                    model = TfLiteModelCreateFromFile("/data/data/com.google.oboe.samples.liveeffect/model_lstm.tflite");
                    options = TfLiteInterpreterOptionsCreate();
                    interpreter = TfLiteInterpreterCreate(model, options);
                    TfLiteInterpreterAllocateTensors(interpreter);
                    input_tensor =
                            TfLiteInterpreterGetInputTensor(interpreter, 0);
                    output_tensor =
                            TfLiteInterpreterGetOutputTensor(interpreter, 0);
                    modelDebugType = 0;
                    ALOG("NN case %d entered",0);
                    break;
                case 1:
                    model = TfLiteModelCreateFromFile("/data/data/com.google.oboe.samples.liveeffect/model_lstm_stateful.tflite");
                    options = TfLiteInterpreterOptionsCreate();
                    interpreter = TfLiteInterpreterCreate(model, options);
                    TfLiteInterpreterAllocateTensors(interpreter);
                    input_tensor =
                            TfLiteInterpreterGetInputTensor(interpreter, 3);
                    hp1_h = TfLiteInterpreterGetInputTensor(interpreter, 6);
                    hp1_c = TfLiteInterpreterGetInputTensor(interpreter, 1);
                    hp2_h = TfLiteInterpreterGetInputTensor(interpreter, 2);
                    hp2_c = TfLiteInterpreterGetInputTensor(interpreter, 5);
                    hp3_h = TfLiteInterpreterGetInputTensor(interpreter, 0);
                    hp3_c = TfLiteInterpreterGetInputTensor(interpreter, 4);

                    output_tensor =
                            TfLiteInterpreterGetOutputTensor(interpreter, 1);
                    hl1_h = TfLiteInterpreterGetOutputTensor(interpreter, 5);
                    hl1_c = TfLiteInterpreterGetOutputTensor(interpreter, 4);
                    hl2_h = TfLiteInterpreterGetOutputTensor(interpreter, 0);
                    hl2_c = TfLiteInterpreterGetOutputTensor(interpreter, 3);
                    hl3_h = TfLiteInterpreterGetOutputTensor(interpreter, 6);
                    hl3_c = TfLiteInterpreterGetOutputTensor(interpreter, 2);

                    modelDebugType = 1;
                    ALOG("NN case %d entered",1);
                    break;
            }
        }
        ALOG("model debug: %d; prev model %d", modelDebugType, prevModelType);
        prevModelType = inputStream->getModelType();


        const clock_t begin_time = clock(); // Start the clock to measure the time of operation

        ALOG("numInputFrames: %d", numInputFrames);
        ALOG("numOutputFrames: %d", numOutputFrames);

        // This code assumes the data format for both streams is Float.
        const float *inputFloats = static_cast<const float *>(inputData); // Neural net input size: 256
        float *outputFloats = static_cast<float *>(outputData); // Neural net output size: 64

        for (int i=0; i<frame_len; ++i)
        {
            sinesig[i] = sin(M_PI*2*f_sig*(i + 64 * part_sine_num) / f_samp);//,4); // arg[1] -> precision
//            sinesig[i] = sin(M_PI*2*f_sig*(i + 64 * part_sine_num) / f_samp); // arg[1] -> precision
            if (part_sine_num >= 25) // 440Hz -> 18.18 próbek na 1 cykl, po przeskalowaniu 1600 próbek na 88 cykli -> 25 przerwań
            {
                part_sine_num = 0;
            }
        }
        part_sine_num++;

//        // 4 x 64 samples to process
//        float *total = new float[256];
//        memcpy(total, prevInputFloats3, 64 * sizeof(float));
//        memcpy(total + 64, prevInputFloats2, 64 * sizeof(float));
//        memcpy(total + 128, prevInputFloats1, 64 * sizeof(float));
//        memcpy(total + 192, inputFloats, 64 * sizeof(float));

        //memcpy(total, sinesig, 256 * sizeof(float)); // Sin synth working

        // 2 x 64 samples to process
        float* total = new float[128];
        // Insert current and previous input into time domain array
        memcpy(total, prevInputFloats1, 64 * sizeof(float));
        memcpy(total + 64, inputFloats, 64 * sizeof(float));

        if (TEST_SINE) {
            for (int i=0; i<frame_len; ++i) {
                total[i] = sinesig[i];
            }
        }

        if (TEST_BUFFER) {
            for (int i=0; i<frame_len; ++i)
            {
                total[i] = loadedBuffer[i];
            }
        }

        if (firstCallback) {
            firstCallback = false;
        }

        if (FOURIER_TFS) {
            if (!hannCoeffsCalculated)
            {
                hann(hannCoeffs, frame_len);
                hannCoeffsCalculated = true;
            }

//            int MAXPATHLEN = 60;
//            char temp[MAXPATHLEN];
//
//            ALOG("cwd: %s", getcwd(temp, sizeof(temp)) ? std::string( temp ).c_str() : std::string("").c_str());
            fftData.open("/sdcard/LiveEffect/fourier_data_logs_testing_buffer.txt", std::ios::app);
            if (!fftData.fail()) {
                fftData << "Callback Start.\n";;
            }
            else {
                ALOG("std::ofstream: write error");
            }

            fftData << "speech signal - Hann window (sample 50)\nbefore:   after:\n";
            // Perform Hann window operations in order to make STFT ops work properly
            for (int i = 0; i < frame_len; i++) {
                complex_sig[i] = hannCoeffs[i] * total[i]; // 0-127
                if (i == 50) // Debug print at sample 50
                {
                    fftData << total[i] << "   ";
                    fftData << complex_sig[i].real() << '\n';
                }
//                complex_sig[i+frame_len/2-1] = hannCoeffs[i] * total[i+frame_len/2-1]; // 128-255

            }

            fftData << "CArray - fft+ifft || complexSig - Hann window\n";

            // Inserting complex array to val array for access to some numerical operations used in fft and ifft
            c_valarray_sig = data_to_c_valarray(complex_sig, frame_len);
//            ALOG("Complex:beforeFTops(%f + i%f)", c_valarray_sig[5].real(),
//                 c_valarray_sig[5].imag());
            fftData << "CArray value checks || samples before fft::\n";
            fftData << "c_valarray_sig[0]: " << c_valarray_sig[0].real() << "\n";
            fftData << "c_valarray_sig[50]: " << c_valarray_sig[50].real() << "\n";
            fftData << "c_valarray_sig[63]: " << c_valarray_sig[63].real() << "\n";
            fft(c_valarray_sig);

            if (TEST_BUFFER)
            {
                fftData << "CArray value checks || All frequencies after fft::\n";
                for (int i=0; i < frame_len; ++i)
                {
                    fftData << "(" << c_valarray_sig[i].real();
                    if (c_valarray_sig[i].imag() > 0)
                    {
                        fftData << "+";
                    }
                    fftData << c_valarray_sig[i].imag() << "j)\n";
                }
            }


            if (NET_PASS)
            {
            // Separating neural net input to two channels, transpose operations
            for (int i = 0; i < frame_len/2 ; ++i)
            {
//                RNNInput[i] = c_valarray_sig[i].real(); // real part of freqs: 0-63 [i]
//                RNNInput[i+64] = c_valarray_sig[i].imag(); // img part of freqs: 0-63 [i+64]
                RNNInput[2*i] = c_valarray_sig[i].real() / 40.0; // / 40.0; // real part of freqs: even [2*i]
                RNNInput[2*i+1] = c_valarray_sig[i].imag() / 40.0; // / 40.0; // img part of freqs: odd [2*i+1]
                // Input size : 2,128
                if (TEST_BUFFER)
                {
                    RNNInput[2*i] = loadedBuffer[2*i];
                    RNNInput[2*i+1] = loadedBuffer[2*i+1];
                }
            }

                TfLiteTensorCopyFromBuffer(
                        input_tensor,
                        RNNInput,
                        TfLiteTensorByteSize(input_tensor));
                if (inputStream->getModelType() == 1)
                {
                    TfLiteTensorCopyFromBuffer(hp1_h,hiddenLayerInputs,TfLiteTensorByteSize(input_tensor));
                    TfLiteTensorCopyFromBuffer(hp1_c,hiddenLayerInputs+128,TfLiteTensorByteSize(input_tensor));
                    TfLiteTensorCopyFromBuffer(hp2_h,hiddenLayerInputs+2*128,TfLiteTensorByteSize(input_tensor));
                    TfLiteTensorCopyFromBuffer(hp2_c,hiddenLayerInputs+3*128,TfLiteTensorByteSize(input_tensor));
                    TfLiteTensorCopyFromBuffer(hp3_h,hiddenLayerInputs+4*128,TfLiteTensorByteSize(input_tensor));
                    TfLiteTensorCopyFromBuffer(hp3_c,hiddenLayerInputs+5*128,TfLiteTensorByteSize(input_tensor));
                }
                TfLiteInterpreterInvoke(interpreter);
                TfLiteTensorCopyToBuffer(
                        output_tensor,
                        outputFloats, //output_data,
                        TfLiteTensorByteSize(output_tensor));
                if (inputStream->getModelType() == 1)
                {
                    TfLiteTensorCopyToBuffer(hl1_h,hiddenLayerOutputs,TfLiteTensorByteSize(output_tensor));
                    TfLiteTensorCopyToBuffer(hl1_c,hiddenLayerOutputs+128,TfLiteTensorByteSize(output_tensor));
                    TfLiteTensorCopyToBuffer(hl2_h,hiddenLayerOutputs+2*128,TfLiteTensorByteSize(output_tensor));
                    TfLiteTensorCopyToBuffer(hl2_c,hiddenLayerOutputs+3*128,TfLiteTensorByteSize(output_tensor));
                    TfLiteTensorCopyToBuffer(hl3_h,hiddenLayerOutputs+4*128,TfLiteTensorByteSize(output_tensor));
                    TfLiteTensorCopyToBuffer(hl3_c,hiddenLayerOutputs+5*128,TfLiteTensorByteSize(output_tensor));

                    // Set current output states to be used in the next callback
                    for (int i = 0; i < 6*128; i++)
                    {
                        hiddenLayerInputs[i] = hiddenLayerOutputs[i];
                    }

                }




//            ALOG("Output Shape %p", *(output_tensor->dims));

            fftData << "Ramka nr" << callbackNum << ":\n[";
            fftData << "Neural Net data IN/OUT:\n";
                if (TEST_BUFFER)
                {
                    for (int i = 0; i < frame_len ; ++i)
                    {
                        fftData << "INP: " << RNNInput[i] << " ";
                        fftData << "OUT: " << outputFloats[i] << "\n";
                    }
                }

            // Output array to CArray conversion
            for (int i = 0; i < frame_len / 2 ; ++i)
            {
                fftData << outputFloats[i] << " ";
                if (i % 4 == 3)
                {
                    fftData << "\n";
                }

                // Compute the masked output
                float re = RNNInput[2*i] * outputFloats[2*i];
                float im = RNNInput[2*i+1] * outputFloats[2*i+1];
//                float re = outputFloats[2*i];
//                float im = outputFloats[2*i+1];
                c_valarray_sig[i] = Complex(re, im); // re, im
            }
                fftData << "]";

            }
//            ALOG("Complex:afterFFTop(%f + i%f)", c_valarray_sig[5].real(),
//                 c_valarray_sig[5].imag());

            ifft(c_valarray_sig);

            fftData << "CArray value checks || samples after fft+ifft::\n";
            fftData << "c_valarray_sig[0]: " << c_valarray_sig[0].real() << "\n";
            fftData << "c_valarray_sig[50]: " << c_valarray_sig[50].real() << "\n";
            fftData << "c_valarray_sig[63]: " << c_valarray_sig[63].real() << "\n";

//            ALOG("Complex:afterFTops(%f + i%f)", c_valarray_sig[5].real(),
//                 c_valarray_sig[5].imag());

//            if (TEST_SINE)
//            {
//                for (int i = 0; i < frame_len; i++) // Write changes back to original sin signal
//                {
//                    sinesig[i] = c_valarray_sig[i].real();
//                }
//            }

//            int count_unchanged_samples = 0;
//            for (int i = 0; i < 128; ++i) {
//
//                fftData << "(" << c_valarray_sig[i].real() << ", " << c_valarray_sig[i].imag() << "i)  || ";
//                fftData << "(" << complex_sig[i].real() << ", " << complex_sig[i].imag() << "i)\n";
//
////                ALOG("Complex:samples changed (%f + i%f)", c_valarray_sig[i].real(),
////                     c_valarray_sig[i].imag());
////            if (i % 16 == 0)
////            {
////                ALOG("sinesigOUT[%d]: %f", i, sinesig[i]);
////            }
//                // Compare base windowed signal with FFT+iFFT results
//                if (float_comp(complex_sig[i].real(), c_valarray_sig[i].real(), 1e-5, 1e-5)) {
//                    count_unchanged_samples++;
//                }
//            }
//            fftData << "%d sample(s) changed after FFT+iFFT operations." << 128-count_unchanged_samples << '\n';
//                ALOG("%d sample(s) changed after FFT+iFFT operations.", 128-count_unchanged_samples);





            outputFloats[50] = c_valarray_sig[50].real() + outputBufferDebug[50];
            outputBufferDebug[50] = c_valarray_sig[50+64].real();

            fftData << "outputFloats[50]: " << outputFloats[50] << "\n";
            fftData << "outputBuffer[50]: " << outputBuffer[50] << "\n";

            fftData << "Callback Continue.\n";
            fftData.close();
        }

//        ALOG("ITERATION: %d", ++iterator);
//        for (int32_t i = 0; i < 256; i++) {
//            ALOG("%d. LAST total %f",i, total[i] );
//        }

        // It also assumes the channel count for each stream is the same.
        int32_t samplesPerFrame = outputStream->getChannelCount();
        int32_t numInputSamples = numInputFrames * samplesPerFrame;
        int32_t numOutputSamples = numOutputFrames * samplesPerFrame;

        // It is possible that there may be fewer input than output samples.
        int32_t samplesToProcess = std::min(numInputSamples, numOutputSamples);

        ALOG("Frame length %d.", samplesToProcess);

        for(int i=0; i < 10; i++)
        {
            ALOG("----> %f", outputFloats[i]);
        }

        // Move sample blocks to previous ones
//        memcpy(prevInputFloats3, prevInputFloats2, 64 * sizeof(float)); // For 4 x 64
//        memcpy(prevInputFloats2, prevInputFloats1, 64 * sizeof(float));
        memcpy(prevInputFloats1, inputFloats, 64 * sizeof(float)); // For 2 x 64

        if (!NET_PASS) // Custom outputs
        {
            for (int i=0; i<64; ++i)
            {
                if (FOURIER_TFS) // Input passed through fourier operations
                {
                    outputFloats[i] = c_valarray_sig[i].real() + outputBuffer[i];
                    outputBuffer[i] = c_valarray_sig[i+64].real(); // outputBuffer is used in the next callback, adding previous to current frame
                }
                else if (TEST_SINE) // Sine input passed to output
                {
                    outputFloats[i] = total[i];
                }
                else // Same input and output
                {
                    outputFloats[i] = inputFloats[i];
                }
            }
        }
        else // Outputs from Neural Net, converted to time domain
        {
            for (int i=0; i<64; ++i)
            {
                outputFloats[i] = c_valarray_sig[i].real() + outputBuffer[i];
                outputBuffer[i] = c_valarray_sig[i+64].real();
            }
        }

        // dB boost
        if (DB_BOOST)
        {
            for (int i=0; i<64; ++i)
            {
                outputFloats[i] *= 10.0;
            }
        }

        // If there are fewer input samples then clear the rest of the buffer.
        int32_t samplesLeft = numOutputSamples - numInputSamples;
        ALOG("Samples left %d.", samplesLeft);

        for (int32_t i = 0; i < samplesLeft; i++) {
            *outputFloats++ = 0.0; // silence
        }

        delete[] total;

        float time_spent = float( clock() - begin_time ) /  CLOCKS_PER_SEC;
        ALOG("Callback time: %f.", time_spent);

        callbackTimeSum += time_spent;
        callbackNum += 1;
        if(callbackNum == 124)
        {
            ALOG("Callback avg time: %f.", callbackTimeSum / (callbackNum + 1));
            callbackTimeSum = 0.0;
            callbackNum = 0;
        }


        return oboe::DataCallbackResult::Continue;
    }
};

#endif //SAMPLES_FULLDUPLEXPASS_H
