package com.example.opticalmarkingrecognition.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = TrustBlue,
    onPrimary = Color.White,
    primaryContainer = TrustBlueLight,
    onPrimaryContainer = CoolGray900,
    secondary = SuccessGreen,
    onSecondary = Color.White,
    secondaryContainer = SuccessGreenLight,
    onSecondaryContainer = CoolGray900,
    tertiary = WarningAmber,
    background = CoolGray,
    onBackground = CoolGray900,
    surface = Color.White,
    onSurface = CoolGray900,
    surfaceVariant = CoolGray100,
    onSurfaceVariant = CoolGray700,
    outline = CoolGray300,
    error = ErrorRed,
    onError = Color.White,
)

private val DarkColorScheme = darkColorScheme(
    primary = TrustBlueLight,
    onPrimary = CoolGray900,
    primaryContainer = TrustBlueDark,
    onPrimaryContainer = Color.White,
    secondary = SuccessGreenLight,
    onSecondary = CoolGray900,
    secondaryContainer = SuccessGreen,
    onSecondaryContainer = Color.White,
    tertiary = WarningAmber,
    background = DarkBackground,
    onBackground = CoolGray100,
    surface = DarkSurface,
    onSurface = CoolGray100,
    surfaceVariant = DarkCard,
    onSurfaceVariant = CoolGray300,
    outline = CoolGray500,
    error = ErrorRed,
    onError = Color.White,
)

@Composable
fun OpticalMarkingRecognitionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}