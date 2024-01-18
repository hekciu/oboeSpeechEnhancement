//
// Created by jakub on 06.11.2023.
//

#ifndef SAMPLES_ONNXHELPER_H
#define SAMPLES_ONNXHELPER_H
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h> // [jlasecki]: It is actually defined so don't worry
#include <algorithm>
#include <cmath>
#include "constants.h"
#include "FourierProcessor.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#ifndef ALOG
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,"test","%s",__VA_ARGS__)
#endif

#ifndef ALOG_NUM
#define ALOG_NUM(...) __android_log_print(ANDROID_LOG_INFO, "test", "%s", std::to_string(__VA_ARGS__).c_str());
#endif

// Inspired by this one: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/Snpe_EP/main.cpp
class OnnxHelper {
private:
    bool _checkStatus(OrtStatus* status) {
        if (status != nullptr) {
            const char* msg = g_ort->GetErrorMessage(status);
            std::cerr << msg << std::endl;
            this->g_ort->ReleaseStatus(status);
            throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
        }
        return true;
    }

    void _fillZeros(float * arr, int numToFill) {
        for (int i = 0; i < numToFill; i++) {
            arr[i] = 0;
        }
    }

    void _fillWithValuesBuffer(float * arr, int arrStart, int valuesBufferStart, int nSamples, const float* valuesBuffer) {
        int j = valuesBufferStart;
        int k = arrStart;

        for (int i = 0; i < nSamples; i++) {
            arr[k] = valuesBuffer[j];
            j++;
            k++;
        }
    }

    void _multiplySamples(float * firstArr, float * secondArr, int N) {
        for (int i = 0; i < N; i++) {
            firstArr[i] = firstArr[i] * secondArr[i];
        }
    }

    void _addSamples(float * firstArr, float * secondArr, int N) {
        for (int i = 0; i < N; i++) {
            firstArr[i] = firstArr[i] + secondArr[i];
        }
    }

    float _degrees_to_radians(float x) {
        return (x / 360) * 2 * this->PI;
    }

    void _createWindow() {
        for (int frameNumber = 0; frameNumber < SAMPLES_TO_MODEL; frameNumber++) {
            float degrees = ((float)frameNumber / (float)SAMPLES_TO_MODEL) * 180;
            float radians = this->_degrees_to_radians(degrees);
            this->window[frameNumber] = sin(radians) * sin(radians);
        }
    }

    OrtEnv* env;
    const OrtApi* g_ort;
    OrtSessionOptions* session_options;
    OrtSession* session;
    OrtAllocator* allocator;
    AAssetManager** mgr;
    AAsset* modelAsset;
    const void * modelDataBuffer;

    float prevSamples[SAMPLES_PER_DATA_CALLBACK];
    float allCurrentSamples[SAMPLES_TO_MODEL];
    float curOutputs[SAMPLES_PER_DATA_CALLBACK];

    float lastOutputToAdd[SAMPLES_PER_DATA_CALLBACK];

    const float PI = 3.14159265359;
    float window[SAMPLES_TO_MODEL];
    float gruHiddenStates[GRU_LAYERS_NUMBER][GRU_HIDDEN_STATE_SIZE];

    FourierProcessor processor;

    float dftInputReal[SAMPLES_TO_MODEL];
    float dftInputImag[SAMPLES_TO_MODEL];

    float idftOutput[SAMPLES_TO_MODEL];
public:
    OnnxHelper(AAssetManager* manager) {
        this->mgr = &manager;

        this->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        this->_checkStatus(this->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &this->env));
        this->_checkStatus(this->g_ort->CreateSessionOptions(&this->session_options));
        this->_checkStatus(this->g_ort->SetIntraOpNumThreads(this->session_options, 1));
        this->_checkStatus(this->g_ort->SetSessionGraphOptimizationLevel(this->session_options, ORT_ENABLE_BASIC));

        this->modelAsset = AAssetManager_open(*this->mgr, "model_something.onnx", AASSET_MODE_BUFFER);

        size_t modelDataBufferLength = (size_t) AAsset_getLength(this->modelAsset);
        this->modelDataBuffer = AAsset_getBuffer(this->modelAsset);
        this->_checkStatus(this->g_ort->CreateSessionFromArray(this->env,
                                                               this->modelDataBuffer,
                                                               modelDataBufferLength,
                                                               this->session_options,
                                                               &this->session));

        this->_checkStatus(this->g_ort->GetAllocatorWithDefaultOptions(&this->allocator));

        this->_fillZeros(this->prevSamples, SAMPLES_PER_DATA_CALLBACK);

        this->_createWindow();

        this->_fillZeros(this->lastOutputToAdd, SAMPLES_PER_DATA_CALLBACK);

        for (size_t n = 0; n < GRU_LAYERS_NUMBER; n++) {
            this->_fillZeros(this->gruHiddenStates[n], GRU_HIDDEN_STATE_SIZE);
        }
    }

    ~OnnxHelper() {
        AAsset_close(this->modelAsset);
        this->g_ort->ReleaseSession(this->session);
        this->g_ort->ReleaseSessionOptions(this->session_options);
        this->g_ort->ReleaseAllocator(this->allocator);
        this->g_ort->ReleaseEnv(this->env);
        delete this->g_ort;
        delete this->mgr;
    }

    void dumbProcessing(float * input, float * output, int N) {
        for (int i = 0; i < N; i++) {
            output[i] = input[i];
        }
    }

    void simpleModelProcessing(const float* input, float * output, size_t numSamples) {
        if (numSamples == 0) {
            ALOG("0 samples, skipping simpleModelProcessing");
            return;
        }

        float actualInput[SAMPLES_PER_DATA_CALLBACK];

        for (int i = 0; i < SAMPLES_PER_DATA_CALLBACK; i++) {
            if (i < numSamples) {
                actualInput[i] = input[i];
            } else {
                actualInput[i] = 0;
            }
        }

        const int64_t shape[] = {(int64_t)numSamples};
        size_t dataLenBytes = shape[0] * sizeof(float);
        size_t shapeLen = 1;


        OrtMemoryInfo * memoryInfo = NULL;
        this->_checkStatus(this->g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo));

        OrtValue * inputTensor = NULL;
        this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                       actualInput,
                                                                       dataLenBytes,
                                                                       shape,
                                                                       shapeLen,
                                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                       &inputTensor));
        OrtValue * outputTensor = NULL;
        this->_checkStatus(this->g_ort->CreateTensorAsOrtValue(this->allocator,
                                                               shape,
                                                               shapeLen,
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                               &outputTensor));

        this->g_ort->ReleaseMemoryInfo(memoryInfo);

        size_t inputCount = 0;
        size_t outputCount = 0;
        this->_checkStatus(this->g_ort->SessionGetInputCount(this->session, &inputCount));
        this->_checkStatus(this->g_ort->SessionGetOutputCount(this->session, &outputCount));

        char * inputNames[1] = {};
        char * outputNames[1] = {};

        for (size_t i = 0; i < inputCount; i++) {
            char * name;
            this->_checkStatus(this->g_ort->SessionGetInputName(this->session,
                                                                i,
                                                                this->allocator,
                                                                &name));
            inputNames[i] = name;
        }

        for (size_t i = 0; i < outputCount; i++) {
            char * name;
            this->_checkStatus(this->g_ort->SessionGetOutputName(this->session,
                                                                i,
                                                                this->allocator,
                                                                &name));
            outputNames[i] = name;
        }

        this->_checkStatus(this->g_ort->Run(this->session,
                                            nullptr,
                                            inputNames,
                                            &inputTensor,
                                            inputCount,
                                            outputNames,
                                            outputCount,
                                            &outputTensor));

        void * buffer = NULL;
        this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensor, &buffer));
        float * floatBuffer = (float *) buffer;

        for (int i = 0; i < SAMPLES_PER_DATA_CALLBACK; i++) {
            *output++ = floatBuffer[i];
        }
//        std::copy(floatBuffer, floatBuffer + shape[0], input);
        this->g_ort->ReleaseValue(inputTensor);
        this->g_ort->ReleaseValue(outputTensor);
    }

    void modelProcessingWithPrevValues(const float * input,
                                       float * output,
                                       size_t numSamples,
                                       bool useWindow = true,
                                       bool useGru = true,
                                       bool useDft = true) {
        if (numSamples == 0) {
            ALOG("0 samples, skipping modelProcessingWithPrevValues");
            return;
        }

        float actualInput[SAMPLES_PER_DATA_CALLBACK];

        for (int i = 0; i < SAMPLES_PER_DATA_CALLBACK; i++) {
            if (i < numSamples) {
                actualInput[i] = input[i];
            } else {
                actualInput[i] = 0;
            }
        }

        // Merging input tensor with prev values
        this->_fillWithValuesBuffer(this->allCurrentSamples, 0, 0, SAMPLES_PER_DATA_CALLBACK, this->prevSamples);
        this->_fillWithValuesBuffer(this->allCurrentSamples, SAMPLES_PER_DATA_CALLBACK, 0, SAMPLES_PER_DATA_CALLBACK, actualInput);

        if (useWindow) {
            this->_multiplySamples(this->allCurrentSamples, this->window, SAMPLES_TO_MODEL);
        }

        int64_t shape[] = { (int64_t) SAMPLES_TO_MODEL };
        size_t dataLenBytes = shape[0] * sizeof(float);
        size_t shapeLen = 1;

        OrtMemoryInfo * memoryInfo = NULL;
        this->_checkStatus(this->g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo));

        OrtValue * inputTensor = NULL;

        this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                       this->allCurrentSamples,
                                                                       dataLenBytes,
                                                                       shape,
                                                                       shapeLen,
                                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                       &inputTensor));

        OrtValue * outputTensor = NULL;

        this->_checkStatus(this->g_ort->CreateTensorAsOrtValue(this->allocator,
                                                               shape,
                                                               shapeLen,
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                               &outputTensor));

        this->g_ort->ReleaseMemoryInfo(memoryInfo);

        size_t inputCount = 0;
        size_t outputCount = 0;
        this->_checkStatus(this->g_ort->SessionGetInputCount(this->session, &inputCount));
        this->_checkStatus(this->g_ort->SessionGetOutputCount(this->session, &outputCount));

        std::vector<char *> inputNames  = {};
        std::vector<char *> outputNames = {};

        for (size_t i = 0; i < inputCount; i++) {
            char * name;
            this->_checkStatus(this->g_ort->SessionGetInputName(this->session,
                                                                i,
                                                                this->allocator,
                                                                &name));
            inputNames.push_back(name);
        }

        for (size_t i = 0; i < outputCount; i++) {
            char * name;
            this->_checkStatus(this->g_ort->SessionGetOutputName(this->session,
                                                                 i,
                                                                 this->allocator,
                                                                 &name));

            outputNames.push_back(name);
        }

        OrtValue * inputTensors[GRU_LAYERS_NUMBER + 1] = {
                inputTensor
        };

        OrtValue * outputTensors[GRU_LAYERS_NUMBER + 1] = {
                outputTensor
        };

        const int GRU_CONV_FACTOR = (GRU_HIDDEN_STATE_SIZE/SAMPLES_TO_MODEL);
        int64_t shapeGru[] = {shape[0] * GRU_CONV_FACTOR};

        if (useGru) {
            for (int n = 1; n < GRU_LAYERS_NUMBER + 1; n++) {
                this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                               this->gruHiddenStates[n-1],
                                                                               dataLenBytes * GRU_CONV_FACTOR,
                                                                               shapeGru,
                                                                               shapeLen,
                                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                               &inputTensors[n]));

                this->_checkStatus(this->g_ort->CreateTensorAsOrtValue(this->allocator,
                                                                       shapeGru,
                                                                       shapeLen,
                                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                       &outputTensors[n]));


            }

            if (useDft) {
                float * dftInput[2] = {
                        this->dftInputReal,
                        this->dftInputImag
                };

                this->processor.dft(this->allCurrentSamples, dftInput);

                OrtValue * inputTensorReal = NULL;
                OrtValue * inputTensorImag = NULL;

                this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                               dftInput[0],
                                                                               dataLenBytes,
                                                                               shape,
                                                                               shapeLen,
                                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                               &inputTensorReal));

                this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                               dftInput[1],
                                                                               dataLenBytes,
                                                                               shape,
                                                                               shapeLen,
                                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                               &inputTensorImag));

                OrtValue * inputTensorsComplex[GRU_LAYERS_NUMBER + 2] = {
                        inputTensorReal,
                        inputTensorImag
                };

                OrtValue * outputTensorReal = NULL;
                OrtValue * outputTensorImag = NULL;

                OrtValue * outputTensorsComplex[GRU_LAYERS_NUMBER + 2] = {
                        outputTensorReal,
                        outputTensorImag
                };

                for (int i = 2; i < GRU_LAYERS_NUMBER + 2; i++) {
                    inputTensorsComplex[i] = inputTensors[i - 1];
                    outputTensorsComplex[i] = outputTensors[i - 1];
                }

                this->_checkStatus(this->g_ort->Run(this->session,
                                                    nullptr,
                                                    inputNames.data(),
                                                    inputTensorsComplex,
                                                    inputCount,
                                                    outputNames.data(),
                                                    outputCount,
                                                    outputTensorsComplex));

                void * realDataBuffer = NULL;
                void * imagDataBuffer = NULL;

                this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensors[0], &realDataBuffer));
                this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensors[1], &imagDataBuffer));

                float * floatRealDataBuffer = (float *) realDataBuffer;
                float * floatImagDataBuffer = (float *) imagDataBuffer;

                float * dftOutput[2] = {
                        floatRealDataBuffer,
                        floatImagDataBuffer
                };

                this->processor.idft(dftOutput, this->idftOutput);

                outputTensors[0] = NULL;

                this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                               this->idftOutput,
                                                                               dataLenBytes,
                                                                               shape,
                                                                               shapeLen,
                                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ,
                                                                               &outputTensors[0]));
            } else {
                this->_checkStatus(this->g_ort->Run(this->session,
                                                    nullptr,
                                                    inputNames.data(),
                                                    inputTensors,
                                                    inputCount,
                                                    outputNames.data(),
                                                    outputCount,
                                                    outputTensors));
            }
        } else {
            if (useDft) {
                throw std::invalid_argument("DFT without GRU not implemented yet");
            } else {
                this->_checkStatus(this->g_ort->Run(this->session,
                                                    nullptr,
                                                    inputNames.data(),
                                                    &inputTensor,
                                                    inputCount,
                                                    outputNames.data(),
                                                    outputCount,
                                                    &outputTensor));
            }
        }

        void * buffer = NULL;
        float * floatBuffer = NULL;

        if (useGru) {
            if (outputCount < GRU_LAYERS_NUMBER + 1) {
                throw std::invalid_argument("Input model doesn't provide enough outputs for GRU hidden states");
            }

            this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensors[0], &buffer));

            floatBuffer = (float *) buffer;

            void * gruStateBuffers[GRU_LAYERS_NUMBER] = {

            };

            for (size_t n = 1; n < GRU_LAYERS_NUMBER + 1; n++) {
                this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensors[n], &gruStateBuffers[n - 1]));

                this->_fillWithValuesBuffer(this->gruHiddenStates[n - 1], 0, 0, GRU_HIDDEN_STATE_SIZE, (float *) gruStateBuffers[n-1]);
            }
        } else {
            this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensor, &buffer));
            floatBuffer = (float *) buffer;
        }

        // Taking last samples from output
        this->_fillWithValuesBuffer(this->curOutputs, 0, 0, SAMPLES_PER_DATA_CALLBACK, floatBuffer);

        if (useWindow) {
            this->_addSamples(this->curOutputs, this->lastOutputToAdd, SAMPLES_PER_DATA_CALLBACK);
            this->_fillWithValuesBuffer(this->lastOutputToAdd, 0, SAMPLES_PER_DATA_CALLBACK, SAMPLES_PER_DATA_CALLBACK, floatBuffer);
        }

//        float * zerosArray = new float[100];
//        this->_fillZeros(zerosArray, 100);

        for (int32_t i = 0; i < SAMPLES_PER_DATA_CALLBACK; i++) {
            *output++ = this->curOutputs[i];
        }
//        this->_fillWithValuesBuffer(*input, 0, 0, SAMPLES_PER_DATA_CALLBACK, zerosArray);
//        std::copy(this->curOutputs, this->curOutputs + SAMPLES_PER_DATA_CALLBACK, input);
        this->g_ort->ReleaseValue(inputTensor);
        this->g_ort->ReleaseValue(outputTensor);

        if (useGru) {
            // TODO release value for complex numbers
            for (int n = 1; n < GRU_LAYERS_NUMBER + 1; n++) {
                this->g_ort->ReleaseValue(inputTensors[n]);
                this->g_ort->ReleaseValue(outputTensors[n]);
            }
        }

        for (int i = 0; i < inputCount; i++) {
            delete inputNames[i];
        }

        for (int i = 0; i < outputCount; i++) {
            delete outputNames[i];
        }

        this->_fillWithValuesBuffer(this->prevSamples, 0,  SAMPLES_PER_DATA_CALLBACK, SAMPLES_PER_DATA_CALLBACK, floatBuffer);
    }
};

#endif //SAMPLES_ONNXHELPER_H
