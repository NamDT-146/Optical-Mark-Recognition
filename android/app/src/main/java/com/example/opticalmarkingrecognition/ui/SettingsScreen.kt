package com.example.opticalmarkingrecognition.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.opticalmarkingrecognition.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen() {
    var exportXlsx by remember { mutableStateOf(true) }
    var exportCsv by remember { mutableStateOf(false) }
    var includeStudentId by remember { mutableStateOf(true) }
    var includeDetails by remember { mutableStateOf(false) }
    var includeTimestamp by remember { mutableStateOf(true) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Settings",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = TrustBlue,
                    titleContentColor = Color.White
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // ── Export Section ──
            SettingsSectionHeader(icon = Icons.Outlined.FileDownload, title = "Export Options")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                elevation = CardDefaults.cardElevation(1.dp)
            ) {
                Column(modifier = Modifier.padding(4.dp)) {
                    SettingsRadioItem(
                        title = "Excel (.XLSX)",
                        selected = exportXlsx,
                        onClick = { exportXlsx = true; exportCsv = false }
                    )
                    HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))
                    SettingsRadioItem(
                        title = "CSV (.CSV)",
                        selected = exportCsv,
                        onClick = { exportCsv = true; exportXlsx = false }
                    )
                }
            }

            // ── Column Mapping ──
            SettingsSectionHeader(icon = Icons.Outlined.ViewColumn, title = "Column Mapping")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                elevation = CardDefaults.cardElevation(1.dp)
            ) {
                Column(modifier = Modifier.padding(4.dp)) {
                    SettingsToggleItem(
                        title = "Include Student ID",
                        checked = includeStudentId,
                        onCheckedChange = { includeStudentId = it }
                    )
                    HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))
                    SettingsToggleItem(
                        title = "Include Question Details",
                        checked = includeDetails,
                        onCheckedChange = { includeDetails = it }
                    )
                    HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))
                    SettingsToggleItem(
                        title = "Include Timestamp",
                        checked = includeTimestamp,
                        onCheckedChange = { includeTimestamp = it }
                    )
                }
            }

            // ── Cloud Section ──
            SettingsSectionHeader(icon = Icons.Outlined.Cloud, title = "Cloud Integration")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                elevation = CardDefaults.cardElevation(1.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Outlined.CloudOff,
                        contentDescription = null,
                        tint = CoolGray500,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Google Drive",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            "Not connected",
                            style = MaterialTheme.typography.bodySmall,
                            color = CoolGray500
                        )
                    }
                    OutlinedButton(
                        onClick = { /* Google Drive integration placeholder */ },
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Connect")
                    }
                }
            }

            // ── About ──
            SettingsSectionHeader(icon = Icons.Outlined.Info, title = "About")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                elevation = CardDefaults.cardElevation(1.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "Smart OMR Grader",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "Version 1.0 • Built with OpenCV + CameraX",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(modifier = Modifier.height(60.dp))
        }
    }
}

@Composable
private fun SettingsSectionHeader(icon: ImageVector, title: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = TrustBlue,
            modifier = Modifier.size(20.dp)
        )
        Text(
            text = title,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = TrustBlue
        )
    }
}

@Composable
private fun SettingsRadioItem(title: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f)
        )
        RadioButton(
            selected = selected,
            onClick = onClick,
            colors = RadioButtonDefaults.colors(selectedColor = TrustBlue)
        )
    }
}

@Composable
private fun SettingsToggleItem(title: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f)
        )
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color.White,
                checkedTrackColor = TrustBlue
            )
        )
    }
}
