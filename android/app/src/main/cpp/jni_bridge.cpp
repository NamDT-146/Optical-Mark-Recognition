#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cstdint>
#include <algorithm>
#include <exception>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#include "grader.hpp"
#include "scanner.hpp"

namespace {

constexpr const char* TAG = "OMR_JNI";

std::string jstringToUtf8(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return "";
    }

    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return "";
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

std::map<int, std::string> parseAnswerKeyCsv(const std::string& csvContent) {
    std::map<int, std::string> answerKey;
    std::istringstream input(csvContent);
    std::string line;
    bool firstRow = true;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream row(line);
        std::string orderStr;
        std::string keyStr;
        std::getline(row, orderStr, ',');
        std::getline(row, keyStr, ',');

        if (firstRow) {
            firstRow = false;
            if (orderStr == "Order" || orderStr == "order") {
                continue;
            }
        }

        orderStr.erase(std::remove(orderStr.begin(), orderStr.end(), '\r'), orderStr.end());
        keyStr.erase(std::remove(keyStr.begin(), keyStr.end(), '\r'), keyStr.end());
        keyStr.erase(std::remove(keyStr.begin(), keyStr.end(), ' '), keyStr.end());

        if (orderStr.empty() || keyStr.empty()) {
            continue;
        }

        try {
            const int order = std::stoi(orderStr);
            answerKey[order] = keyStr;
        } catch (...) {
            // Ignore malformed rows and keep parsing.
        }
    }

    return answerKey;
}

std::string buildScoreSummary(const GradingResult& parsed, const std::map<int, std::string>& answerKey) {
    int correct = 0;
    int wrong = 0;
    int blank = 0;
    int multiple = 0;

    // --- NEW: Print student ID and Exam code to the console ---
    __android_log_print(ANDROID_LOG_INFO, TAG, "Parsed Student ID: %s", parsed.student_id.c_str());
    __android_log_print(ANDROID_LOG_INFO, TAG, "Parsed Exam Code: %s", parsed.exam_code.c_str());
    __android_log_print(ANDROID_LOG_INFO, TAG, "--- Parsed Answers ---");

    // --- NEW: Iterate and print all parsed answers ---
    for (const auto& questionPair : parsed.questions) {
        int questionNum = questionPair.first;
        std::string parsedAnswer = questionPair.second;
        
        // Output something like "Question 1: A"
        __android_log_print(ANDROID_LOG_INFO, TAG, "Question %d: %s", 
                            questionNum, 
                            parsedAnswer.empty() ? "[BLANK]" : parsedAnswer.c_str());
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "------------------------");

    for (const auto& keyPair : answerKey) {
        const int question = keyPair.first;
        const std::string& expected = keyPair.second;

        const auto parsedIt = parsed.questions.find(question);
        const std::string parsedAnswer = (parsedIt != parsed.questions.end()) ? parsedIt->second : "";

        if (parsedAnswer.empty()) {
            blank++;
            continue;
        }

        if (parsedAnswer == "M") {
            multiple++;
            wrong++;
            continue;
        }

        if (parsedAnswer == expected) {
            correct++;
        } else {
            wrong++;
        }
    }

    const int total = static_cast<int>(answerKey.size());
    const double score10 = (total > 0) ? (static_cast<double>(correct) / static_cast<double>(total)) * 10.0 : 0.0;

    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    output << "OMR_OK"
        << "|opencv=" << CV_VERSION
        << "|exam_code=" << parsed.exam_code
        << "|student_id=" << parsed.student_id
        << "|correct=" << correct << "/" << total
        << "|wrong=" << wrong
        << "|blank=" << blank
        << "|multiple=" << multiple
        << "|score10=" << score10;

    // --- NEW: Add all parsed answers to the output string ---
    output << "|answers=[";
    bool first = true;
    for (const auto& questionPair : parsed.questions) {
        if (!first) output << ",";
        output << questionPair.first << ":" << (questionPair.second.empty() ? "BLANK" : questionPair.second);
        first = false;
    }
    output << "]";

    return output.str();
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_opticalmarkingrecognition_NativeEngine_processImage(
        JNIEnv* env, jobject /*thiz*/, jobject bitmap, jstring answerKeyCsv) {
    if (bitmap == nullptr) {
        return env->NewStringUTF("JNI_ERR: bitmap is null");
    }

    const std::map<int, std::string> parsedAnswerKey = parseAnswerKeyCsv(jstringToUtf8(env, answerKeyCsv));
    if (parsedAnswerKey.empty()) {
        return env->NewStringUTF("JNI_ERR: answer key CSV is empty or invalid");
    }

    AndroidBitmapInfo info{};
    const int info_result = AndroidBitmap_getInfo(env, bitmap, &info);
    if (info_result != ANDROID_BITMAP_RESULT_SUCCESS) {
        return env->NewStringUTF("JNI_ERR: AndroidBitmap_getInfo failed");
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return env->NewStringUTF("JNI_ERR: bitmap format must be RGBA_8888");
    }

    void* pixels = nullptr;
    bool isLocked = false;

    const auto unlockPixels = [&]() {
        if (isLocked) {
            AndroidBitmap_unlockPixels(env, bitmap);
            isLocked = false;
        }
    };

    const int lock_result = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (lock_result != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) {
        return env->NewStringUTF("JNI_ERR: AndroidBitmap_lockPixels failed");
    }
    isLocked = true;

    try {
        const cv::Mat rgba(
                static_cast<int>(info.height),
                static_cast<int>(info.width),
                CV_8UC4,
                pixels,
                static_cast<size_t>(info.stride)
        );

        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        unlockPixels();

        if (bgr.empty()) {
            return env->NewStringUTF("JNI_ERR: converted BGR image is empty");
        }

        const omr::ScanWarpResult scan_result = omr::scanAndWarpOmrPage(
                bgr,
                cv::Size(2100, 2970),
                false
        );
        if (!scan_result.success) {
            const std::string error = std::string("JNI_ERR: scan/warp failed: ") + scan_result.error;
            return env->NewStringUTF(error.c_str());
        }

        __android_log_print(ANDROID_LOG_INFO, TAG, "OpenCV version: %s", CV_VERSION);

        AnswerParser parser;
        const GradingResult parsed = parser.parse_answers(scan_result.warped, "");
        const std::string response = buildScoreSummary(parsed, parsedAnswerKey);
        return env->NewStringUTF(response.c_str());
    } catch (const cv::Exception& e) {
        unlockPixels();
        const std::string error = std::string("JNI_ERR: OpenCV exception: ") + e.what();
        return env->NewStringUTF(error.c_str());
    } catch (const std::exception& e) {
        unlockPixels();
        const std::string error = std::string("JNI_ERR: exception: ") + e.what();
        return env->NewStringUTF(error.c_str());
    } catch (...) {
        unlockPixels();
        return env->NewStringUTF("JNI_ERR: unknown native exception");
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_opticalmarkingrecognition_NativeEngine_processLiveFrame(
        JNIEnv* env, jobject /*thiz*/, 
        jobject byteBuffer, jint width, jint height, jint rowStride, jstring answerKeyCsv) {

    if (byteBuffer == nullptr) {
        return env->NewStringUTF("JNI_ERR: ByteBuffer is null");
    }

    uint8_t* pData = (uint8_t*)env->GetDirectBufferAddress(byteBuffer);
    if (!pData) {
        return env->NewStringUTF("JNI_ERR: Cannot get DirectBufferAddress");
    }

    try {
        // Create grayscale cv::Mat using the Y plane data
        cv::Mat grayFrame(height, width, CV_8UC1, pData, rowStride);

        // Camera frames are usually rotated 90 degrees clockwise on mobile
        cv::Mat rotatedFrame;
        cv::rotate(grayFrame, rotatedFrame, cv::ROTATE_90_CLOCKWISE);

        // Since our pipeline expects BGR (because scanAndWarpOmrPage uses cvtColor BGR2GRAY internally and looks for color),
        // we should convert the grayscale frame to BGR to reuse the existing pipeline without changes.
        // Or, we can modify scanAndWarpOmrPage to handle 1-channel images, but let's just convert it here for simplicity and safety.
        cv::Mat bgrFrame;
        cv::cvtColor(rotatedFrame, bgrFrame, cv::COLOR_GRAY2BGR);

        // Process just like processImage
        const omr::ScanWarpResult scan_result = omr::scanAndWarpOmrPage(
                bgrFrame,
                cv::Size(2100, 2970),
                false
        );

        if (!scan_result.success) {
            return env->NewStringUTF("JNI_WARN: No document found");
        }

        const std::map<int, std::string> parsedAnswerKey = parseAnswerKeyCsv(jstringToUtf8(env, answerKeyCsv));
        if (parsedAnswerKey.empty()) {
             return env->NewStringUTF("JNI_ERR: answer key CSV is empty or invalid");
        }

        AnswerParser parser;
        const GradingResult parsed = parser.parse_answers(scan_result.warped, "");
        const std::string response = buildScoreSummary(parsed, parsedAnswerKey);
        
        return env->NewStringUTF(response.c_str());

    } catch (const std::exception& e) {
        const std::string error = std::string("JNI_ERR: exception: ") + e.what();
        return env->NewStringUTF(error.c_str());
    } catch (...) {
        return env->NewStringUTF("JNI_ERR: unknown native exception");
    }
}