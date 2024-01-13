//
// Created by jakub on 21.11.2023.
//

#ifndef SAMPLES_CONSTANTS_H
#define SAMPLES_CONSTANTS_H

const int SAMPLES_PER_DATA_CALLBACK = 100;
const int SAMPLES_TO_MODEL = 2 * SAMPLES_PER_DATA_CALLBACK;
const int SAMPLE_RATE = 8000;
//const int FFT_N = 202;
const int GRU_LAYERS_NUMBER = 3;

//const char * USER_MODEL_PATH = "model_gru.onnx";

#endif //SAMPLES_CONSTANTS_H
