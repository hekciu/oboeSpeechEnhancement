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

#ifndef ALOG
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,"test",__VA_ARGS__)
#endif

#include "fileHelper.h"
#include "onnxHelper.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

class FullDuplexPass : public oboe::FullDuplexStream {
private:
    OnnxHelper* onnxHelper;
    FileHelper* fileHelper;
public:
    FullDuplexPass(AAssetManager* manager) {
        this->onnxHelper = new OnnxHelper(manager);
        this->fileHelper = new FileHelper(manager); //chyba z tym managerem jednak zly pomysl
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

//        ALOG("n_samples: %d", numOutputSamples);

        // It is possible that there may be fewer input than output samples.
        int32_t samplesToProcess = std::min(numInputSamples, numOutputSamples);
        for (int32_t i = 0; i < samplesToProcess; i++) {
//            *outputFloats++ = *inputFloats++ * 0.95; // do some arbitrary processing
            *outputFloats++ = *inputFloats++;
        }

        // If there are fewer input samples then clear the rest of the buffer.
        int32_t samplesLeft = numOutputSamples - numInputSamples;
        for (int32_t i = 0; i < samplesLeft; i++) {
            *outputFloats++ = 0.0; // silence
        }

//        FileHelper fileHelper = FileHelper();
//        fileHelper.saveValue((double)numOutputSamples, "/data/data/LiveEffect/numOutputSamples.txt");

        this->onnxHelper->simpleModelProcessing(outputFloats, numOutputSamples);
//        this->onnxHelper->modelProcessingWithPrevValues(outputFloats, numOutputSamples);

        return oboe::DataCallbackResult::Continue;
    }
};
#endif //SAMPLES_FULLDUPLEXPASS_H
