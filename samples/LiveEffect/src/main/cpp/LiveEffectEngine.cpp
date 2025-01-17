/**
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

#include <cassert>
#include <logging_macros.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "LiveEffectEngine.h"
#include "FullDuplexPass.h"

#ifndef ALOG
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,"test",__VA_ARGS__)
#endif

LiveEffectEngine::LiveEffectEngine(AAssetManager* manager):
        mFullDuplexPass(manager) {

    assert(mOutputChannelCount == mInputChannelCount);
}

void LiveEffectEngine::setRecordingDeviceId(int32_t deviceId) {
    mRecordingDeviceId = deviceId;
}

void LiveEffectEngine::setPlaybackDeviceId(int32_t deviceId) {
    mPlaybackDeviceId = deviceId;
}

bool LiveEffectEngine::isAAudioRecommended() {
    return oboe::AudioStreamBuilder::isAAudioRecommended();
}

bool LiveEffectEngine::setAudioApi(oboe::AudioApi api) {
    if (mIsEffectOn) return false;
    mAudioApi = api;
    return true;
}

bool LiveEffectEngine::setEffectOn(bool isOn) {
    bool success = true;
    if (isOn != mIsEffectOn) {
        if (isOn) {
            success = openStreams() == oboe::Result::OK;
            if (success) {
                mFullDuplexPass.start();
                mIsEffectOn = isOn;
            }
        } else {
            mFullDuplexPass.stop();
            closeStreams();
            mIsEffectOn = isOn;
       }
    }
    return success;
}

void LiveEffectEngine::closeStreams() {
    /*
    * Note: The order of events is important here.
    * The playback stream must be closed before the recording stream. If the
    * recording stream were to be closed first the playback stream's
    * callback may attempt to read from the recording stream
    * which would cause the app to crash since the recording stream would be
    * null.
    */
    closeStream(mPlayStream);
    mFullDuplexPass.setOutputStream(nullptr);

    closeStream(mRecordingStream);
    mFullDuplexPass.setInputStream(nullptr);
}

oboe::Result  LiveEffectEngine::openStreams() {
    // Note: The order of stream creation is important. We create the playback
    // stream first, then use properties from the playback stream
    // (e.g. sample rate) to create the recording stream. By matching the
    // properties we should get the lowest latency path
    oboe::AudioStreamBuilder inBuilder, outBuilder;
    setupPlaybackStreamParameters(&outBuilder);
    oboe::Result result = outBuilder.openStream(mPlayStream);
    if (result != oboe::Result::OK) {
        LOGE("Failed to open output stream. Error %s", oboe::convertToText(result));
        mSampleRate = oboe::kUnspecified;
        return result;
    } else {
        // The input stream needs to run at the same sample rate as the output.
        mSampleRate = mPlayStream->getSampleRate();
    }
    warnIfNotLowLatency(mPlayStream);

    setupRecordingStreamParameters(&inBuilder, mSampleRate);
    result = inBuilder.openStream(mRecordingStream);
    if (result != oboe::Result::OK) {
        LOGE("Failed to open input stream. Error %s", oboe::convertToText(result));
        closeStream(mPlayStream);
        return result;
    }
    warnIfNotLowLatency(mRecordingStream);

    mFullDuplexPass.setInputStream(mRecordingStream.get());
    mFullDuplexPass.setOutputStream(mPlayStream.get());
    return result;
}

/**
 * Sets the stream parameters which are specific to recording,
 * including the sample rate which is determined from the
 * playback stream.
 *
 * @param builder The recording stream builder
 * @param sampleRate The desired sample rate of the recording stream
 */
oboe::AudioStreamBuilder *LiveEffectEngine::setupRecordingStreamParameters(
    oboe::AudioStreamBuilder *builder, int32_t sampleRate) {
    // This sample uses blocking read() because we don't specify a callback
    builder->setDeviceId(mRecordingDeviceId)
        ->setDirection(oboe::Direction::Input)
        ->setFramesPerDataCallback(SAMPLES_PER_DATA_CALLBACK)
        ->setSampleRate(SAMPLE_RATE)
        ->setChannelCount(mInputChannelCount);

    return setupCommonStreamParameters(builder);
}

/**
 * Sets the stream parameters which are specific to playback, including device
 * id and the dataCallback function, which must be set for low latency
 * playback.
 * @param builder The playback stream builder
 */
oboe::AudioStreamBuilder *LiveEffectEngine::setupPlaybackStreamParameters(
    oboe::AudioStreamBuilder *builder) {
    builder->setDataCallback(this)
        ->setErrorCallback(this)
        ->setDeviceId(mPlaybackDeviceId)
        ->setDirection(oboe::Direction::Output)
        ->setChannelCount(mOutputChannelCount);

    return setupCommonStreamParameters(builder);
}

/**
 * Set the stream parameters which are common to both recording and playback
 * streams.
 * @param builder The playback or recording stream builder
 */
oboe::AudioStreamBuilder *LiveEffectEngine::setupCommonStreamParameters(
    oboe::AudioStreamBuilder *builder) {
    // We request EXCLUSIVE mode since this will give us the lowest possible
    // latency.
    // If EXCLUSIVE mode isn't available the builder will fall back to SHARED
    // mode.
    builder->setAudioApi(mAudioApi)
        ->setFormat(mFormat)
        ->setFormatConversionAllowed(true)
        ->setSharingMode(oboe::SharingMode::Exclusive)
        ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
        ->setFramesPerDataCallback(SAMPLES_PER_DATA_CALLBACK)
        ->setSampleRate(SAMPLE_RATE);
    return builder;
}

/**
 * Close the stream. AudioStream::close() is a blocking call so
 * the application does not need to add synchronization between
 * onAudioReady() function and the thread calling close().
 * [the closing thread is the UI thread in this sample].
 * @param stream the stream to close
 */
void LiveEffectEngine::closeStream(std::shared_ptr<oboe::AudioStream> &stream) {
    if (stream) {
        oboe::Result result = stream->stop();
        if (result != oboe::Result::OK) {
            LOGW("Error stopping stream: %s", oboe::convertToText(result));
        }
        result = stream->close();
        if (result != oboe::Result::OK) {
            LOGE("Error closing stream: %s", oboe::convertToText(result));
        } else {
            LOGW("Successfully closed streams");
        }
        stream.reset();
    }
}

/**
 * Warn in logcat if non-low latency stream is created
 * @param stream: newly created stream
 *
 */
void LiveEffectEngine::warnIfNotLowLatency(std::shared_ptr<oboe::AudioStream> &stream) {
    if (stream->getPerformanceMode() != oboe::PerformanceMode::LowLatency) {
        LOGW(
            "Stream is NOT low latency."
            "Check your requested format, sample rate and channel count");
    }
}

/**
 * Handles playback stream's audio request. In this sample, we simply block-read
 * from the record stream for the required samples.
 *
 * @param oboeStream: the playback stream that requesting additional samples
 * @param audioData:  the buffer to load audio samples for playback stream
 * @param numFrames:  number of frames to load to audioData buffer
 * @return: DataCallbackResult::Continue.
 */
oboe::DataCallbackResult LiveEffectEngine::onAudioReady(
    oboe::AudioStream *oboeStream, void *audioData, int32_t numFrames) {
    return mFullDuplexPass.onAudioReady(oboeStream, audioData, numFrames);
}

/**
 * Oboe notifies the application for "about to close the stream".
 *
 * @param oboeStream: the stream to close
 * @param error: oboe's reason for closing the stream
 */
void LiveEffectEngine::onErrorBeforeClose(oboe::AudioStream *oboeStream,
                                          oboe::Result error) {
    LOGE("%s stream Error before close: %s",
         oboe::convertToText(oboeStream->getDirection()),
         oboe::convertToText(error));
}

/**
 * Oboe notifies application that "the stream is closed"
 *
 * @param oboeStream
 * @param error
 */
void LiveEffectEngine::onErrorAfterClose(oboe::AudioStream *oboeStream,
                                         oboe::Result error) {
    LOGE("%s stream Error after close: %s",
         oboe::convertToText(oboeStream->getDirection()),
         oboe::convertToText(error));

    // Stop the Full Duplex stream.
    // Since the error callback occurs only for the output stream, close the input stream.
    mFullDuplexPass.stop();
    mFullDuplexPass.setOutputStream(nullptr);
    closeStream(mRecordingStream);
    mFullDuplexPass.setInputStream(nullptr);

    // Restart the stream if the error is a disconnect.
    if (error == oboe::Result::ErrorDisconnected) {
        LOGI("Restarting AudioStream");
        oboe::Result result = openStreams();
        if (result == oboe::Result::OK) {
            mFullDuplexPass.start();
        }
    }
}
