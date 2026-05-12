package com.example.opticalmarkingrecognition.ui

import androidx.compose.animation.*
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.opticalmarkingrecognition.database.ClassEntity
import com.example.opticalmarkingrecognition.ui.theme.*
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    classes: List<ClassEntity>,
    onClassClick: (Int) -> Unit,
    onAddClass: (String) -> Unit,
    onDeleteClass: (ClassEntity) -> Unit,
    onScanClick: () -> Unit
) {
    var showAddDialog by remember { mutableStateOf(false) }
    var newClassName by remember { mutableStateOf("") }

    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onScanClick,
                icon = { Icon(Icons.Filled.CameraAlt, contentDescription = "Quick Scan") },
                text = { Text("Quick Scan") },
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
            // ── Header ──
            item {
                DashboardHeader(totalClasses = classes.size)
            }

            // ── Quick Stats ──
            item {
                QuickStatsRow(totalClasses = classes.size)
            }

            // ── Classes Section ──
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Your Classes",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    FilledTonalIconButton(
                        onClick = { showAddDialog = true },
                        colors = IconButtonDefaults.filledTonalIconButtonColors(
                            containerColor = TrustBlue.copy(alpha = 0.1f),
                            contentColor = TrustBlue
                        )
                    ) {
                        Icon(Icons.Filled.Add, contentDescription = "Add Class")
                    }
                }
            }

            if (classes.isEmpty()) {
                item {
                    EmptyStateCard(
                        icon = Icons.Outlined.School,
                        title = "No Classes Yet",
                        subtitle = "Tap + to create your first class"
                    )
                }
            } else {
                items(classes, key = { it.id }) { classEntity ->
                    ClassCard(
                        classEntity = classEntity,
                        onClick = { onClassClick(classEntity.id) },
                        onDelete = { onDeleteClass(classEntity) }
                    )
                }
            }
        }
    }

    // ── Add Class Dialog ──
    if (showAddDialog) {
        AlertDialog(
            onDismissRequest = { showAddDialog = false },
            icon = { Icon(Icons.Outlined.School, contentDescription = null, tint = TrustBlue) },
            title = { Text("Create New Class") },
            text = {
                OutlinedTextField(
                    value = newClassName,
                    onValueChange = { newClassName = it },
                    label = { Text("Class Name") },
                    placeholder = { Text("e.g. Computer Science 101") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = TrustBlue,
                        focusedLabelColor = TrustBlue,
                        cursorColor = TrustBlue
                    )
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (newClassName.isNotBlank()) {
                            onAddClass(newClassName.trim())
                            newClassName = ""
                            showAddDialog = false
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = TrustBlue)
                ) {
                    Text("Create")
                }
            },
            dismissButton = {
                TextButton(onClick = { showAddDialog = false; newClassName = "" }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun DashboardHeader(totalClasses: Int) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    colors = listOf(TrustBlue, TrustBlueDark)
                )
            )
            .padding(horizontal = 20.dp, vertical = 28.dp)
    ) {
        Column {
            Text(
                text = getGreeting(),
                style = MaterialTheme.typography.titleSmall,
                color = Color.White.copy(alpha = 0.8f)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Smart OMR Grader",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "$totalClasses active class${if (totalClasses != 1) "es" else ""}",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun QuickStatsRow(totalClasses: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        StatMiniCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.School,
            value = "$totalClasses",
            label = "Classes",
            color = TrustBlue
        )
        StatMiniCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.DocumentScanner,
            value = "—",
            label = "Scanned",
            color = SuccessGreen
        )
        StatMiniCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.CloudDone,
            value = "—",
            label = "Synced",
            color = WarningAmber
        )
    }
}

@Composable
private fun StatMiniCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    value: String,
    label: String,
    color: Color
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.08f)
        ),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
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
private fun ClassCard(
    classEntity: ClassEntity,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteConfirm by remember { mutableStateOf(false) }
    val dateFormat = remember { SimpleDateFormat("MMM dd, yyyy", Locale.getDefault()) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 6.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Icon
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(TrustBlue.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Outlined.School,
                    contentDescription = null,
                    tint = TrustBlue,
                    modifier = Modifier.size(24.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            // Text
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = classEntity.name,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Created ${dateFormat.format(Date(classEntity.createdAt))}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Actions
            IconButton(onClick = { showDeleteConfirm = true }) {
                Icon(
                    Icons.Outlined.Delete,
                    contentDescription = "Delete",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(20.dp)
                )
            }

            Icon(
                Icons.Filled.ChevronRight,
                contentDescription = "Open",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
        }
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            icon = { Icon(Icons.Outlined.Warning, contentDescription = null, tint = ErrorRed) },
            title = { Text("Delete Class?") },
            text = { Text("This will permanently delete \"${classEntity.name}\" and all its exams and results.") },
            confirmButton = {
                Button(
                    onClick = { onDelete(); showDeleteConfirm = false },
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed)
                ) {
                    Text("Delete")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
fun EmptyStateCard(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 6.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        ),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
            )
        }
    }
}

private fun getGreeting(): String {
    val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
    return when {
        hour < 12 -> "Good Morning 👋"
        hour < 17 -> "Good Afternoon 👋"
        else -> "Good Evening 👋"
    }
}
