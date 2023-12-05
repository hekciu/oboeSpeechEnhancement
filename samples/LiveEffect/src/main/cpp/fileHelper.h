//
// Created by jakub on 20.11.2023.
//

#ifndef SAMPLES_FILEHELPER_H
#define SAMPLES_FILEHELPER_H

#ifndef ALOG
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,"test",__VA_ARGS__)
#endif

#include <string>
#include <fstream>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

class FileHelper {
private:
    AAssetManager** mgr;
public:
    FileHelper(AAssetManager* manager) {
        this->mgr = &manager;
    };
    ~FileHelper() {};

    void saveValue(double input, std::string path) {
        std::string msg = std::to_string(input);
        std::fstream file;
        file.open(path, std::fstream::in | std::fstream::out | std::fstream::app);
        file << msg;
        file.close();
    }
};

#endif //SAMPLES_FILEHELPER_H
