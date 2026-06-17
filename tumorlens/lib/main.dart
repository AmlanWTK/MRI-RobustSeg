import 'package:brain_tumor_detection/screens/home_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const MyApp());
}

// ─── Design System Tokens ────────────────────────────────────────────────────
class AppColors {
  // Backgrounds
  static const Color bg           = Color(0xFF060B14);
  static const Color bgSurface    = Color(0xFF0D1421);
  static const Color bgCard       = Color(0xFF111927);
  static const Color bgCardHover  = Color(0xFF16202E);

  // Accent — Cyan-Teal medical glow
  static const Color accentPrimary    = Color(0xFF00D4FF);
  static const Color accentSecondary  = Color(0xFF0087CC);
  static const Color accentGlow       = Color(0xFF00AAFF);
  static const Color accentSoft       = Color(0xFF003D5C);

  // Status
  static const Color danger    = Color(0xFFFF4D6D);
  static const Color warning   = Color(0xFFFFB347);
  static const Color success   = Color(0xFF00E5A0);
  static const Color info      = Color(0xFF7EB8FF);

  // Text
  static const Color textPrimary   = Color(0xFFE8F4FF);
  static const Color textSecondary = Color(0xFF6B8CAE);
  static const Color textMuted     = Color(0xFF3A5068);

  // Borders / dividers
  static const Color border      = Color(0xFF1C2D3F);
  static const Color borderGlow  = Color(0xFF00AAFF);

  // Tumor types
  static const Color enhancing = Color(0xFFFF4D6D);
  static const Color core      = Color(0xFF7EB8FF);
  static const Color whole     = Color(0xFF00E5A0);
}

class AppTheme {
  static ThemeData get dark => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.bg,
    fontFamily: 'Roboto',
    colorScheme: const ColorScheme.dark(
      primary: AppColors.accentPrimary,
      secondary: AppColors.accentSecondary,
      surface: AppColors.bgSurface,
      error: AppColors.danger,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.transparent,
      foregroundColor: AppColors.textPrimary,
      elevation: 0,
      scrolledUnderElevation: 0,
      titleTextStyle: TextStyle(
        color: AppColors.textPrimary,
        fontSize: 18,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.5,
      ),
      iconTheme: IconThemeData(color: AppColors.accentPrimary),
    ),
    cardTheme: CardThemeData(
      color: AppColors.bgCard,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: AppColors.border, width: 1),
      ),
    ),
    textTheme: const TextTheme(
      displayLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.bold),
      headlineMedium: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w700),
      titleLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
      bodyLarge: TextStyle(color: AppColors.textSecondary),
      bodyMedium: TextStyle(color: AppColors.textSecondary),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.accentPrimary,
        foregroundColor: AppColors.bg,
        elevation: 0,
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, letterSpacing: 0.5),
      ),
    ),
    dividerTheme: const DividerThemeData(color: AppColors.border, thickness: 1),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: AppColors.bgCard,
      contentTextStyle: TextStyle(color: AppColors.textPrimary),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AIService()),
      ],
      child: MaterialApp(
        title: 'TumorLens — AI Brain Analysis',
        theme: AppTheme.dark,
        home: const HomeScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
