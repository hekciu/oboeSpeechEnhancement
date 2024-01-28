//
// Created by jakub on 21.11.2023.
//

#ifndef SAMPLES_CONSTANTS_H
#define SAMPLES_CONSTANTS_H

const int SAMPLES_PER_DATA_CALLBACK = 64;
const int SAMPLES_TO_MODEL = 2 * SAMPLES_PER_DATA_CALLBACK;
const int SAMPLE_RATE = 8192;
//const int FFT_N = 202;
const int GRU_LAYERS_NUMBER = 3;
//const int GRU_HIDDEN_STATE_SIZE = 256; // 400
const int GRU_HIDDEN_STATE_SIZES[] = {128, 128, 256};

//const char * USER_MODEL_PATH = "model_gru.onnx";

#endif //SAMPLES_CONSTANTS_H
