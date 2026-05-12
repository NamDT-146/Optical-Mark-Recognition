package com.example.opticalmarkingrecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.opticalmarkingrecognition.database.ClassEntity
import com.example.opticalmarkingrecognition.ui.*
import com.example.opticalmarkingrecognition.ui.theme.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            OpticalMarkingRecognitionTheme {
                OMRApp()
            }
        }
    }
}

// ── Navigation destinations ──
sealed class AppScreen {
    data object Dashboard : AppScreen()
    data class ExamList(val classId: Int, val className: String) : AppScreen()
    data class ExamDetail(val examId: Int) : AppScreen()
    data object Camera : AppScreen()
    data object Settings : AppScreen()
}

enum class BottomNavItem(val label: String, val icon: @Composable () -> Unit) {
    Dashboard("Dashboard", { Icon(Icons.Outlined.Dashboard, contentDescription = "Dashboard") }),
    Scan("Scan", { Icon(Icons.Filled.CameraAlt, contentDescription = "Scan") }),
    Settings("Settings", { Icon(Icons.Outlined.Settings, contentDescription = "Settings") })
}

@Composable
fun OMRApp() {
    val viewModel: OMRViewModel = viewModel()
    var currentScreen by remember { mutableStateOf<AppScreen>(AppScreen.Dashboard) }
    var selectedNavItem by remember { mutableIntStateOf(0) }

    // Collect state from ViewModel
    val classes by viewModel.allClasses.collectAsState()
    val examsForClass by viewModel.examsForSelectedClass.collectAsState()
    val resultsForExam by viewModel.resultsForSelectedExam.collectAsState()
    val examStats by viewModel.examStats.collectAsState()
    val selectedExam by viewModel.selectedExam.collectAsState()

    // Show bottom nav only on main screens
    val showBottomNav = currentScreen is AppScreen.Dashboard ||
            currentScreen is AppScreen.Settings

    Scaffold(
        bottomBar = {
            if (showBottomNav) {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surface,
                    tonalElevation = 8.dp
                ) {
                    BottomNavItem.entries.forEachIndexed { index, item ->
                        NavigationBarItem(
                            selected = selectedNavItem == index,
                            onClick = {
                                selectedNavItem = index
                                currentScreen = when (item) {
                                    BottomNavItem.Dashboard -> AppScreen.Dashboard
                                    BottomNavItem.Scan -> AppScreen.Camera
                                    BottomNavItem.Settings -> AppScreen.Settings
                                }
                            },
                            icon = item.icon,
                            label = {
                                Text(
                                    item.label,
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = if (selectedNavItem == index) FontWeight.Bold else FontWeight.Normal
                                )
                            },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = TrustBlue,
                                selectedTextColor = TrustBlue,
                                indicatorColor = TrustBlue.copy(alpha = 0.1f),
                                unselectedIconColor = CoolGray500,
                                unselectedTextColor = CoolGray500
                            )
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding)) {
            when (val screen = currentScreen) {
                is AppScreen.Dashboard -> {
                    DashboardScreen(
                        classes = classes,
                        onClassClick = { classId ->
                            val className = classes.find { it.id == classId }?.name ?: "Class"
                            viewModel.selectClass(classId)
                            currentScreen = AppScreen.ExamList(classId, className)
                        },
                        onAddClass = { name -> viewModel.addClass(name) },
                        onDeleteClass = { entity -> viewModel.deleteClass(entity) },
                        onScanClick = {
                            selectedNavItem = 1
                            currentScreen = AppScreen.Camera
                        }
                    )
                }

                is AppScreen.ExamList -> {
                    ExamListScreen(
                        className = screen.className,
                        exams = examsForClass,
                        onExamClick = { examId ->
                            viewModel.selectExam(examId)
                            currentScreen = AppScreen.ExamDetail(examId)
                        },
                        onAddExam = { name, totalQuestions ->
                            viewModel.addExam(screen.classId, name, totalQuestions)
                        },
                        onDeleteExam = { exam -> viewModel.deleteExam(exam) },
                        onBack = {
                            viewModel.clearClassSelection()
                            currentScreen = AppScreen.Dashboard
                        }
                    )
                }

                is AppScreen.ExamDetail -> {
                    ExamDetailScreen(
                        exam = selectedExam,
                        stats = examStats,
                        results = resultsForExam,
                        onStartScoring = {
                            currentScreen = AppScreen.Camera
                        },
                        onBack = {
                            viewModel.clearExamSelection()
                            val classId = viewModel.selectedClassId.value
                            if (classId != null) {
                                val className = classes.find { it.id == classId }?.name ?: "Class"
                                viewModel.selectClass(classId)
                                currentScreen = AppScreen.ExamList(classId, className)
                            } else {
                                currentScreen = AppScreen.Dashboard
                            }
                        }
                    )
                }

                is AppScreen.Camera -> {
                    CameraScreenWrapper(
                        onBack = {
                            selectedNavItem = 0
                            currentScreen = AppScreen.Dashboard
                        }
                    )
                }

                is AppScreen.Settings -> {
                    SettingsScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CameraScreenWrapper(onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "OMR Scanner",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = TrustBlue,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                )
            )
        }
    ) { innerPadding ->
        CameraScreen(modifier = Modifier.padding(innerPadding))
    }
}

private fun decodeArgb8888Bitmap(context: Context, uri: Uri): Bitmap? {
    return runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            val decoded = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
            decoded.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            @Suppress("DEPRECATION")
            val decoded = MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            decoded.copy(Bitmap.Config.ARGB_8888, true)
        }
    }.getOrNull()
}

private fun readAssetText(context: Context, fileName: String): String {
    return runCatching {
        context.assets.open(fileName).bufferedReader().use { it.readText() }
    }.getOrDefault("")
}