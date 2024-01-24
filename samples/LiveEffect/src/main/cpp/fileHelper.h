//
// Created by jakub on 24.01.2024.
//

#ifndef SAMPLES_FILEHELPER_H
#define SAMPLES_FILEHELPER_H

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>


class FileHelper {
private:
    AAssetManager** mgr;
    AAsset* assetInput;
    AAsset* assetOutput;
public:
    FileHelper(AAssetManager * manager) {
        this->mgr = &manager;
        if (!this->mgr) {
            throw std::invalid_argument("Error loading AAssetManager");
        }
        this->assetInput = AAssetManager_open(*this->mgr, "7_samples.txt", AASSET_MODE_BUFFER);
        this->assetOutput = AAssetManager_open(*this->mgr, "output_test_samples.txt", AASSET_MODE_UNKNOWN);
    }

    ~FileHelper() {
        AAsset_close(this->assetInput);
        AAsset_close(this->assetOutput);
        delete this->mgr;
    }

//    void saveFloatBufferToTxt(float * inputBuffer, int N = 256) {
//        std::string output = "";
//
//        for (int i = 0; i < N; i++) {
//            output += std::to_string(inputBuffer[i]) + '\n';
//        }
//    }

    void getFloatBufferFromTxt(float * outputBuffer) {
        if (this->assetInput == NULL) {
            throw std::invalid_argument("Wrong file path");
        }

        long size = AAsset_getLength(this->assetInput);
        char* buffer = (char*) malloc (sizeof(char)*size);
        AAsset_read(this->assetInput, buffer, size);
        AAsset_close(this->assetInput);

        int currCharCounter = 0;
        char * curString = new char[10];
        int floatsCounter = 0;
        for (int i = 0; i < size; i++) {
            if(buffer[i * sizeof(char)] == '\n') {

                char * str = new char[currCharCounter];

                for (int j = 0; j < currCharCounter; j++) {
                    str[j] = curString[j];
                }

                float newFloat = atof(str);
                outputBuffer[floatsCounter] = newFloat;
                floatsCounter++;
                currCharCounter = 0;

                delete[] str;
            } else {
                curString[currCharCounter] = char(int(buffer[i * sizeof(char)]));
                currCharCounter++;
            }
        }
    }
};

#endif //SAMPLES_FILEHELPER_H
