package com.example.opticalmarkingrecognition

import android.graphics.Bitmap

object NativeEngine {
    init {
        System.loadLibrary("opencv_java4")
        System.loadLibrary("omrcore")
    }

    external fun processImage(bitmap: Bitmap, answerKeyCsv: String): String

    external fun processLiveFrame(
        byteBuffer: java.nio.ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        answerKeyCsv: String
    ): String
}
