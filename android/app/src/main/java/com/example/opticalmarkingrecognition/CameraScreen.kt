package com.example.opticalmarkingrecognition

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@Composable
fun CameraScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            hasCameraPermission = granted
        }
    )

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasCameraPermission) {
        CameraPreview(modifier = modifier)
    } else {
        Column(
            modifier = modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("Camera permission is required")
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Grant Permission")
            }
        }
    }
}

@Composable
fun CameraPreview(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    
    var resultText by remember { mutableStateOf("Point camera at OMR sheet") }
    var answerKeyCsv by remember { mutableStateOf("") }
    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isCapturing by remember { mutableStateOf(false) }
    var isCameraPaused by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var captureRequested by remember { mutableStateOf(false) }
    
    LaunchedEffect(Unit) {
        answerKeyCsv = readAssetText(context, "TEST_1.csv")
    }

    if (capturedBitmap != null && isCameraPaused) {
        // Show captured image and analysis options
        Column(
            modifier = modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text("Captured Image")
            
            capturedBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Captured image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    contentScale = ContentScale.Fit
                )
            }

            Button(
                onClick = {
                    if (answerKeyCsv.isNotBlank() && capturedBitmap != null) {
                        isProcessing = true
                        coroutineScope.launch {
                            val nativeResult = withContext(Dispatchers.Default) {
                                runCatching {
                                    NativeEngine.processImage(capturedBitmap!!, answerKeyCsv)
                                }.getOrElse { error ->
                                    "JNI_ERR: ${error.message ?: "Unknown error"}"
                                }
                            }
                            resultText = nativeResult
                            Log.i("OMR_CAPTURE", "Analysis result: $nativeResult")
                            isProcessing = false
                        }
                    }
                },
                enabled = !isProcessing
            ) {
                Text(if (isProcessing) "Analyzing..." else "Analyze")
            }

            Button(
                onClick = {
                    isCameraPaused = false
                    capturedBitmap = null
                    resultText = "Point camera at OMR sheet"
                }
            ) {
                Text("Retake")
            }

            if (isProcessing) {
                CircularProgressIndicator()
            }

            Text("Result: $resultText")
        }
    } else {
        // Show camera preview
        Column(modifier = modifier.fillMaxSize()) {
            AndroidView(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        this.scaleType = PreviewView.ScaleType.FILL_CENTER
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                    }

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()

                        val executor = Executors.newSingleThreadExecutor()
                        imageAnalysis.setAnalyzer(executor) { imageProxy ->
                            try {
                                // Capture frame if requested
                                if (captureRequested && !isCameraPaused) {
                                    val bitmap = imageProxyToBitmap(imageProxy)
                                    capturedBitmap = bitmap
                                    isCameraPaused = true
                                    captureRequested = false
                                }

                                // Show live results only if not paused
                                if (!isCameraPaused && answerKeyCsv.isNotBlank() && imageProxy.planes.isNotEmpty()) {
                                    val planeProxy = imageProxy.planes[0]
                                    val buffer = planeProxy.buffer
                                    
                                    val result = NativeEngine.processLiveFrame(
                                        buffer,
                                        imageProxy.width,
                                        imageProxy.height,
                                        planeProxy.rowStride,
                                        answerKeyCsv
                                    )
                                    
                                    if (!result.startsWith("JNI_WARN")) {
                                        resultText = result
                                        Log.i("OMR_LIVE", "Result: $result")
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e("OMR_LIVE", "Error processing frame", e)
                            } finally {
                                imageProxy.close()
                            }
                        }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            Log.e("CameraPreview", "Binding failed", e)
                        }

                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                }
            )
            
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(text = resultText)
            }

            Button(
                onClick = { captureRequested = true },
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(16.dp)
            ) {
                Text("Capture & Analyze")
            }
        }
    }
}

private fun imageProxyToBitmap(imageProxy: androidx.camera.core.ImageProxy): Bitmap {
    val planes = imageProxy.planes
    val yPlane = planes[0]
    val uvPixelStride = planes[1].pixelStride

    val ySize = yPlane.buffer.remaining()
    val uvSize = planes[1].buffer.remaining()

    val nv21 = ByteArray(ySize + uvSize)

    yPlane.buffer.get(nv21, 0, ySize)
    val uvBuffer = planes[1].buffer
    val uvPixelStride_final = uvPixelStride
    if (uvPixelStride_final == 1) {
        uvBuffer.get(nv21, ySize, uvSize)
    } else {
        val uvData = ByteArray(uvSize)
        uvBuffer.get(uvData)
        for (i in 0 until uvSize step 2) {
            nv21[ySize + i] = uvData[i]
            nv21[ySize + i + 1] = uvData[i]
        }
    }

    val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
    val yPixelStride = yPlane.pixelStride
    if (yPixelStride == 1) {
        nv21ToBitmap(nv21, imageProxy.width, imageProxy.height, bitmap)
    }

    return bitmap
}

private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int, bitmap: Bitmap) {
    val pixels = IntArray(width * height)
    var pixelIndex = 0

    for (i in 0 until height) {
        for (j in 0 until width) {
            val y = (nv21[pixelIndex].toInt() and 0xff)
            val u = (nv21[width * height + 2 * (i / 2) + 1].toInt() and 0xff)
            val v = (nv21[width * height + 2 * (i / 2)].toInt() and 0xff)

            pixels[pixelIndex] = convertYUVtoRGB(y, u, v)
            pixelIndex++
        }
    }

    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
}

private fun convertYUVtoRGB(y: Int, u: Int, v: Int): Int {
    val yF = y.toFloat()
    val uF = u - 128f
    val vF = v - 128f

    val r = (yF + 1.402f * vF).toInt().coerceIn(0, 255)
    val g = (yF - 0.344f * uF - 0.714f * vF).toInt().coerceIn(0, 255)
    val b = (yF + 1.772f * uF).toInt().coerceIn(0, 255)

    return (0xff shl 24) or (r shl 16) or (g shl 8) or b
}

private fun readAssetText(context: Context, fileName: String): String {
    return runCatching {
        context.assets.open(fileName).bufferedReader().use { it.readText() }
    }.getOrDefault("")
}