package com.example.opticalmarkingrecognition.ui

import androidx.compose.foundation.background
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
import com.example.opticalmarkingrecognition.database.ExamStatsData
import com.example.opticalmarkingrecognition.database.ScannedResultEntity
import com.example.opticalmarkingrecognition.ui.theme.*
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExamDetailScreen(
    exam: ExamEntity?,
    stats: ExamStatsData,
    results: List<ScannedResultEntity>,
    onStartScoring: () -> Unit,
    onBack: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            exam?.name ?: "Exam Details",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        exam?.let {
                            Text(
                                "${it.totalQuestions} questions",
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.White.copy(alpha = 0.7f)
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { /* Export placeholder */ }) {
                        Icon(Icons.Outlined.FileDownload, contentDescription = "Export")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = TrustBlue,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White,
                    actionIconContentColor = Color.White
                )
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onStartScoring,
                icon = { Icon(Icons.Filled.CameraAlt, contentDescription = "Start Scoring") },
                text = { Text("Start Scoring") },
                containerColor = SuccessGreen,
                contentColor = Color.White,
                shape = RoundedCornerShape(16.dp)
            )
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(bottom = 80.dp)
        ) {
            // ── Analytics Overview ──
            item {
                AnalyticsOverview(
                    totalScanned = results.size,
                    avgScore = stats.avgScore,
                    maxScore = stats.maxScore,
                    totalQuestions = exam?.totalQuestions ?: 0
                )
            }

            // ── Results Header ──
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Scanned Results",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    if (results.isNotEmpty()) {
                        Surface(
                            shape = RoundedCornerShape(20.dp),
                            color = TrustBlue.copy(alpha = 0.1f)
                        ) {
                            Text(
                                text = "${results.size} students",
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Medium,
                                color = TrustBlue
                            )
                        }
                    }
                }
            }

            if (results.isEmpty()) {
                item {
                    EmptyStateCard(
                        icon = Icons.Outlined.DocumentScanner,
                        title = "No Results Yet",
                        subtitle = "Tap \"Start Scoring\" to begin scanning"
                    )
                }
            } else {
                items(results, key = { it.id }) { result ->
                    ResultCard(result = result, totalQuestions = exam?.totalQuestions ?: 0)
                }
            }
        }
    }
}

@Composable
private fun AnalyticsOverview(
    totalScanned: Int,
    avgScore: Double?,
    maxScore: Int?,
    totalQuestions: Int
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        AnalyticsCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.People,
            value = "$totalScanned",
            label = "Total Scanned",
            color = TrustBlue
        )
        AnalyticsCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.TrendingUp,
            value = if (avgScore != null) String.format("%.1f", avgScore) else "—",
            label = "Average",
            color = WarningAmber
        )
        AnalyticsCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.EmojiEvents,
            value = if (maxScore != null) "$maxScore/$totalQuestions" else "—",
            label = "Highest",
            color = SuccessGreen
        )
    }
}

@Composable
private fun AnalyticsCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    value: String,
    label: String,
    color: Color
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(color.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ResultCard(result: ScannedResultEntity, totalQuestions: Int) {
    val dateFormat = remember { SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault()) }
    val scorePercent = if (totalQuestions > 0) (result.rawScore.toFloat() / totalQuestions * 100) else 0f
    val scoreColor = when {
        scorePercent >= 80 -> SuccessGreen
        scorePercent >= 60 -> WarningAmber
        else -> ErrorRed
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 4.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Student icon
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(scoreColor.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Outlined.Person,
                    contentDescription = null,
                    tint = scoreColor,
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            // Student info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "ID: ${result.studentId}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Code: ${result.examCode}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = "•",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = dateFormat.format(Date(result.scannedAt)),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Score
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = "${result.rawScore}/${result.maxScore}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = scoreColor
                )
                // Verified badge
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (result.isVerified) SuccessGreen.copy(alpha = 0.1f) else WarningAmber.copy(alpha = 0.1f)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            if (result.isVerified) Icons.Filled.CheckCircle else Icons.Filled.Warning,
                            contentDescription = null,
                            modifier = Modifier.size(12.dp),
                            tint = if (result.isVerified) SuccessGreen else WarningAmber
                        )
                        Text(
                            text = if (result.isVerified) "Verified" else "Review",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (result.isVerified) SuccessGreen else WarningAmber,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }
        }
    }
}
