//
// Created by jakub on 14.12.2023.
//

#ifndef SAMPLES_ANDROIDLOGGER_H
#define SAMPLES_ANDROIDLOGGER_H

//https://medium.com/@geierconstantinabc/best-way-to-log-in-android-native-code-c-style-7461005610f6

//class AndroidLogger{
//public:
//    AndroidLogger(const android_LogPriority priority,const std::string& TAG):M_PRIORITY(priority),M_TAG(TAG) {}
//    ~AndroidLogger() {
//        __android_log_print(ANDROID_LOG_DEBUG,M_TAG.c_str(),"%s",stream.str().c_str());
//    }
//private:
//    std::stringstream stream;
//    const std::string M_TAG;
//    const android_LogPriority M_PRIORITY;
//    template <typename T>
//    friend AndroidLogger& operator<<(AndroidLogger& record, T&& t);
//};
//template <typename T>
//AndroidLogger& operator<<(AndroidLogger& record, T&& t) {
//    record.stream << std::forward<T>(t);
//    return record;
//}

#endif //SAMPLES_ANDROIDLOGGER_H
