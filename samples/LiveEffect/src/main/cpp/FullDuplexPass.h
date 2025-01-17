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

#include "onnxHelper.h"
#include "fileHelper.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

class FullDuplexPass : public oboe::FullDuplexStream {
private:
    OnnxHelper<float> * onnxHelper;
    FileHelper * fileHelper;
//    int debugCounter = 0;

public:
    FullDuplexPass(AAssetManager* manager) {
        this->onnxHelper = new OnnxHelper<float>(manager);
        this->fileHelper = new FileHelper(manager);
    }
    ~FullDuplexPass() {
        delete this->onnxHelper;
    }
    virtual oboe::DataCallbackResult
    onBothStreamsReady(
            const void *inputData,
            int   numInputFrames,
            void *outputData,
            int   numOutputFrames) {
        // Copy the input samples to the output with a little arbitrary gain change.

        // This code assumes the data format for both streams is Float.

        const float *inputFloats = static_cast<const float *>(inputData);
        float *outputFloats = static_cast<float *>(outputData);

        // It also assumes the channel count for each stream is the same.
        int32_t samplesPerFrame = getOutputStream()->getChannelCount();
        int32_t numInputSamples = numInputFrames * samplesPerFrame;
        int32_t numOutputSamples = numOutputFrames * samplesPerFrame;

        ALOG_NUM(numInputSamples);
        ALOG_NUM(numOutputSamples);

//        ALOG("n_samples: %d", numOutputSamples);

        // It is possible that there may be fewer input than output samples.
//        int32_t samplesToProcess = std::min(numInputSamples, numOutputSamples);
//        for (int32_t i = 0; i < samplesToProcess; i++) {
////            *outputFloats++ = *inputFloats++ * 0.95; // do some arbitrary processing
//            *outputFloats++ = *inputFloats++;
//        }

        // If there are fewer input samples then clear the rest of the buffer.
//        int32_t samplesLeft = numOutputSamples - numInputSamples;
//        for (int32_t i = 0; i < samplesLeft; i++) {
//            *outputFloats++ = 0.0; // silence
//        }

//        for (int i = 0; i < 100; i++) {
//            outputFloats[i] = 0;
//        }

//        this->onnxHelper->simpleModelProcessing(inputFloats, outputFloats, numOutputSamples);

//        float * buff = new float[256];
//        float * outputBuff = new float[256];

//        this->fileHelper->getFloatBufferFromTxt(buff);

//        float * inputFloats = new float[128];
//
//        for (int i = 0; i < 128; i++) {
//            inputFloats[i] = buff[i];
//        }

        this->onnxHelper->modelProcessingWithPrevValues(inputFloats,
        outputFloats,
        numInputSamples,
        true,
        true,
        true);

//        this->onnxHelper->processDebug(inputFloats, outputFloats);
//
//        this->debugCounter++;
//
//        if (debugCounter == 400) {
//            std::string test = "";
//            for (int i = 50; i < 80; i++) {
//                test += "; in: " + std::to_string(inputFloats[i]) + " out: " + std::to_string(outputFloats[i]);
//            }
//
//            throw std::invalid_argument(test);
//        }

//        this->onnxHelper->doOnlyFourierProcessing(buff, outputBuff);

//        this->onnxHelper->doOnlyWindow(buff, outputBuff);
//

        return oboe::DataCallbackResult::Continue;
    }
};
#endif //SAMPLES_FULLDUPLEXPASS_H
