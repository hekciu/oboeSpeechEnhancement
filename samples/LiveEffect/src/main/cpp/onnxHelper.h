//
// Created by jakub on 06.11.2023.
//

#ifndef SAMPLES_ONNXHELPER_H
#define SAMPLES_ONNXHELPER_H
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h> // [jlasecki]: It is actually defined so don't worry
#include <algorithm>
#include "constants.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#ifndef ORT_API_VERSION
#define ORT_API_VERSION   17
#endif

#ifndef ALOG
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,"test",__VA_ARGS__)
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

    void _fillZeros(float* arr, int numToFill) {
        for (int i = 0; i < numToFill; i++) {
            arr[i] = 0;
        }
    }

    void _fillWithValuesBuffer(float* arr, int arrStart, int valuesBufferStart, int numToFill, float* valuesBuffer) {
        int j = valuesBufferStart;

        for (int i = arrStart; i < numToFill; i++) {
            arr[i] = valuesBuffer[j];
            j++;
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
    float prevSamples[FRAMES_PER_DATA_CALLBACK];
    float allCurrentSamples[FRAMES_PER_DATA_CALLBACK * 2];
    float curOutputs[FRAMES_PER_DATA_CALLBACK];

public:
    OnnxHelper(AAssetManager* manager) {
        this->mgr = &manager;

        this->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        this->_checkStatus(this->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &this->env));
        this->_checkStatus(this->g_ort->CreateSessionOptions(&this->session_options));
        this->_checkStatus(this->g_ort->SetIntraOpNumThreads(this->session_options, 1));
        this->_checkStatus(this->g_ort->SetSessionGraphOptimizationLevel(this->session_options, ORT_ENABLE_BASIC));

        const char* model_path = "model.onnx";
        this->modelAsset = AAssetManager_open(*this->mgr, model_path, AASSET_MODE_BUFFER);

        std::vector<const char*> options_keys = {"runtime"};
        std::vector<const char*> options_values = {"CPU"};

//        this->_checkStatus(this->g_ort->SessionOptionsAppendExecutionProvider(this->session_options, "SNPE", options_keys.data(),
//                                                     options_values.data(), options_keys.size()));

        size_t modelDataBufferLength = (size_t) AAsset_getLength(this->modelAsset);
        this->modelDataBuffer = AAsset_getBuffer(this->modelAsset);
        this->_checkStatus(this->g_ort->CreateSessionFromArray(this->env,
                                                               this->modelDataBuffer,
                                                               modelDataBufferLength,
                                                               this->session_options,
                                                               &this->session));

        this->_checkStatus(this->g_ort->GetAllocatorWithDefaultOptions(&this->allocator));

        this->_fillZeros(this->prevSamples, FRAMES_PER_DATA_CALLBACK);
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



    float* dumbProcessing(float* input) {
        return input;
    }

    void simpleModelProcessing(float* input, size_t numSamples) {
        if (numSamples == 0) {
            ALOG("0 samples, skipping simpleModelProcessing");
            return;
        }

        const int64_t shape[] = {(int64_t)numSamples};
        size_t dataLenBytes = shape[0] * sizeof(float);
        size_t shapeLen = 1;


        OrtMemoryInfo * memoryInfo = NULL;
        this->_checkStatus(this->g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo));

        OrtValue * inputTensor = NULL;
        this->_checkStatus(this->g_ort->CreateTensorWithDataAsOrtValue(memoryInfo,
                                                                       input,
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

        void * buffer = NULL;;
        this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensor, &buffer));
        float * floatBuffer = (float *) buffer;
        std::copy(floatBuffer, floatBuffer + shape[0], input);
        this->g_ort->ReleaseValue(inputTensor);
        this->g_ort->ReleaseValue(outputTensor);
    }

    void modelProcessingWithPrevValues(float* input, size_t numSamples) {
        if (numSamples == 0) {
            ALOG("0 samples, skipping simpleModelProcessing");
            return;
        }

        // Merging input tensor with prev values
        this->_fillWithValuesBuffer(this->allCurrentSamples, 0, 0, FRAMES_PER_DATA_CALLBACK, this->prevSamples);
        this->_fillWithValuesBuffer(this->allCurrentSamples, FRAMES_PER_DATA_CALLBACK, 0, 2 * FRAMES_PER_DATA_CALLBACK, input);

        const int64_t shape[] = {(int64_t)numSamples * 2}; // We will be using new merged samples array
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

        void * buffer = NULL;;
        this->_checkStatus(this->g_ort->GetTensorMutableData(outputTensor, &buffer));
        float * floatBuffer = (float *) buffer;

        // Taking only 'current' samples from output
        this->_fillWithValuesBuffer(this->curOutputs, 0, FRAMES_PER_DATA_CALLBACK, FRAMES_PER_DATA_CALLBACK, floatBuffer);

        std::copy(this->curOutputs, this->curOutputs + FRAMES_PER_DATA_CALLBACK, input);
        this->g_ort->ReleaseValue(inputTensor);
        this->g_ort->ReleaseValue(outputTensor);

        this->_fillWithValuesBuffer(this->prevSamples, 0,  FRAMES_PER_DATA_CALLBACK, FRAMES_PER_DATA_CALLBACK, floatBuffer);
    }
};

#endif //SAMPLES_ONNXHELPER_H
