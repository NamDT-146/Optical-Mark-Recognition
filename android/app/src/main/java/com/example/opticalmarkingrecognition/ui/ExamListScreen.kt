package com.example.opticalmarkingrecognition.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.opticalmarkingrecognition.database.ExamEntity
import com.example.opticalmarkingrecognition.database.ExamWithStats
import com.example.opticalmarkingrecognition.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExamListScreen(
    className: String,
    exams: List<ExamWithStats>,
    onExamClick: (Int) -> Unit,
    onAddExam: (name: String, totalQuestions: Int) -> Unit,
    onDeleteExam: (ExamEntity) -> Unit,
    onBack: () -> Unit
) {
    var showAddDialog by remember { mutableStateOf(false) }
    var newExamName by remember { mutableStateOf("") }
    var newExamQuestions by remember { mutableStateOf("40") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            className,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            "${exams.size} exam${if (exams.size != 1) "s" else ""}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
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
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { showAddDialog = true },
                icon = { Icon(Icons.Filled.Add, contentDescription = "Add Exam") },
                text = { Text("New Exam") },
                containerColor = TrustBlue,
                contentColor = Color.White,
                shape = RoundedCornerShape(16.dp)
            )
        }
    ) { innerPadding ->
        if (exams.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
                contentAlignment = Alignment.Center
            ) {
                EmptyStateCard(
                    icon = Icons.Outlined.Assignment,
                    title = "No Exams Yet",
                    subtitle = "Create your first exam to start scoring"
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
                contentPadding = PaddingValues(vertical = 12.dp, horizontal = 0.dp)
            ) {
                items(exams, key = { it.exam.id }) { examWithStats ->
                    ExamCard(
                        examWithStats = examWithStats,
                        onClick = { onExamClick(examWithStats.exam.id) },
                        onDelete = { onDeleteExam(examWithStats.exam) }
                    )
                }
            }
        }
    }

    // ── Add Exam Dialog ──
    if (showAddDialog) {
        AlertDialog(
            onDismissRequest = { showAddDialog = false },
            icon = { Icon(Icons.Outlined.Assignment, contentDescription = null, tint = TrustBlue) },
            title = { Text("Create New Exam") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = newExamName,
                        onValueChange = { newExamName = it },
                        label = { Text("Exam Name") },
                        placeholder = { Text("e.g. Midterm Exam") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = TrustBlue,
                            focusedLabelColor = TrustBlue,
                            cursorColor = TrustBlue
                        )
                    )
                    OutlinedTextField(
                        value = newExamQuestions,
                        onValueChange = { newExamQuestions = it.filter { c -> c.isDigit() } },
                        label = { Text("Total Questions") },
                        placeholder = { Text("40") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = TrustBlue,
                            focusedLabelColor = TrustBlue,
                            cursorColor = TrustBlue
                        )
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val questions = newExamQuestions.toIntOrNull() ?: 40
                        if (newExamName.isNotBlank()) {
                            onAddExam(newExamName.trim(), questions)
                            newExamName = ""
                            newExamQuestions = "40"
                            showAddDialog = false
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = TrustBlue)
                ) {
                    Text("Create")
                }
            },
            dismissButton = {
                TextButton(onClick = { showAddDialog = false; newExamName = ""; newExamQuestions = "40" }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun ExamCard(
    examWithStats: ExamWithStats,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val exam = examWithStats.exam
    val scannedCount = examWithStats.scannedCount
    var showDeleteConfirm by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 6.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Icon
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(SuccessGreen.copy(alpha = 0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Outlined.Assignment,
                        contentDescription = null,
                        tint = SuccessGreen,
                        modifier = Modifier.size(22.dp)
                    )
                }

                Spacer(modifier = Modifier.width(12.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = exam.name,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "${exam.totalQuestions} questions",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                // Scanned badge
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = if (scannedCount > 0) SuccessGreen.copy(alpha = 0.1f) else CoolGray200,
                    contentColor = if (scannedCount > 0) SuccessGreen else CoolGray500
                ) {
                    Text(
                        text = "Scanned: $scannedCount",
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Medium
                    )
                }

                IconButton(onClick = { showDeleteConfirm = true }) {
                    Icon(
                        Icons.Outlined.Delete,
                        contentDescription = "Delete",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            // Progress indicator
            if (scannedCount > 0) {
                Spacer(modifier = Modifier.height(12.dp))
                LinearProgressIndicator(
                    progress = { (scannedCount.toFloat() / 50f).coerceAtMost(1f) }, // Assume ~50 students per class
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp)),
                    color = SuccessGreen,
                    trackColor = SuccessGreen.copy(alpha = 0.1f)
                )
            }
        }
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            icon = { Icon(Icons.Outlined.Warning, contentDescription = null, tint = ErrorRed) },
            title = { Text("Delete Exam?") },
            text = { Text("This will permanently delete \"${exam.name}\" and all scanned results.") },
            confirmButton = {
                Button(
                    onClick = { onDelete(); showDeleteConfirm = false },
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed)
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) { Text("Cancel") }
            }
        )
    }
}
