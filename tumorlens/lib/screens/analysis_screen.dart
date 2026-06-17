// ignore_for_file: prefer_const_constructors

import 'dart:math' as math;

import 'package:brain_tumor_detection/main.dart';
import 'package:brain_tumor_detection/models/tumor_result.dart';
import 'package:brain_tumor_detection/screens/result_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:typed_data';


// ─── Analysis Screen ──────────────────────────────────────────────────────────
class AnalysisScreen extends StatefulWidget {
  final List<Uint8List> imageBytes;

  const AnalysisScreen({Key? key, required this.imageBytes}) : super(key: key);

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> with TickerProviderStateMixin {
  late AnimationController _scanController;
  late AnimationController _pulseController;
  late AnimationController _orbitController;
  late Animation<double> _scanAnim;
  late Animation<double> _pulseAnim;
  late Animation<double> _orbitAnim;

  bool _analysisComplete = false;
  TumorResult? _result;
  String? _errorMessage;

  final List<String> _steps = [
    'Initializing neural network...',
    'Preprocessing MRI volumes...',
    'Running encoder pathway...',
    'Attention gate computation...',
    'Decoder reconstruction...',
    'Generating segmentation maps...',
    'Computing confidence metrics...',
    'Finalizing clinical report...',
  ];
  int _currentStep = 0;

  @override
  void initState() {
    super.initState();

    _scanController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);

    _orbitController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 5),
    )..repeat();

    _scanAnim = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _scanController, curve: Curves.linear),
    );
    _pulseAnim = Tween<double>(begin: 0.7, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    _orbitAnim = Tween<double>(begin: 0.0, end: 2 * math.pi).animate(
      CurvedAnimation(parent: _orbitController, curve: Curves.linear),
    );

    _startAnalysis();
    _stepTimer();
  }

  void _stepTimer() async {
    for (int i = 0; i < _steps.length; i++) {
      await Future.delayed(const Duration(milliseconds: 800));
      if (mounted && !_analysisComplete) {
        setState(() => _currentStep = i);
      }
    }
  }

  @override
  void dispose() {
    _scanController.dispose();
    _pulseController.dispose();
    _orbitController.dispose();
    super.dispose();
  }

  Future<void> _startAnalysis() async {
    try {
      final aiService = Provider.of<AIService>(context, listen: false);
      // Wait for the next frame to avoid setState during build issues
      await Future.delayed(Duration.zero);
      final result = await aiService.analyzeMRI(widget.imageBytes);

      setState(() {
        _result = result;
        _analysisComplete = true;
      });

      _scanController.stop();
      _pulseController.stop();
      _orbitController.stop();

      await Future.delayed(const Duration(milliseconds: 800));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (_, anim, __) => ResultsScreen(result: result),
            transitionsBuilder: (_, anim, __, child) => FadeTransition(opacity: anim, child: child),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _analysisComplete = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              // Top cancel
              if (!_analysisComplete && _errorMessage == null)
                Align(
                  alignment: Alignment.centerRight,
                  child: GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                      decoration: BoxDecoration(
                        color: AppColors.bgCard,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: const Text(
                        'Cancel',
                        style: TextStyle(color: AppColors.textSecondary, fontSize: 13),
                      ),
                    ),
                  ),
                ),

              const Spacer(),

              if (_errorMessage != null)
                _buildErrorState()
              else if (_analysisComplete)
                _buildSuccessState()
              else
                _buildScanningState(),

              const Spacer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScanningState() {
    return Column(
      children: [
        // Animated scanner orb
        SizedBox(
          width: 220,
          height: 220,
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Orbit ring
              AnimatedBuilder(
                animation: _orbitAnim,
                builder: (_, __) => Transform.rotate(
                  angle: _orbitAnim.value,
                  child: Container(
                    width: 200,
                    height: 200,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppColors.accentPrimary.withOpacity(0.15),
                        width: 1,
                      ),
                    ),
                    child: Stack(
                      children: [
                        Positioned(
                          top: 0,
                          left: 90,
                          child: Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: AppColors.accentPrimary,
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(color: AppColors.accentPrimary, blurRadius: 8, spreadRadius: 2),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              // Second orbit (slower, opposite)
              AnimatedBuilder(
                animation: _orbitAnim,
                builder: (_, __) => Transform.rotate(
                  angle: -_orbitAnim.value * 0.7,
                  child: Container(
                    width: 160,
                    height: 160,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppColors.info.withOpacity(0.1),
                        width: 1,
                      ),
                    ),
                    child: Stack(
                      children: [
                        Positioned(
                          bottom: 0,
                          right: 68,
                          child: Container(
                            width: 6,
                            height: 6,
                            decoration: BoxDecoration(
                              color: AppColors.info,
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(color: AppColors.info, blurRadius: 6),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              // Scan line
              AnimatedBuilder(
                animation: _scanAnim,
                builder: (_, __) {
                  final y = (_scanAnim.value * 100) - 50;
                  return Transform.translate(
                    offset: Offset(0, y),
                    child: Container(
                      width: 100,
                      height: 2,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Colors.transparent,
                            AppColors.accentPrimary.withOpacity(0.6),
                            Colors.transparent,
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
              // Center pulsing brain
              AnimatedBuilder(
                animation: _pulseAnim,
                builder: (_, __) => Transform.scale(
                  scale: _pulseAnim.value,
                  child: Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: RadialGradient(
                        colors: [
                          AppColors.accentPrimary.withOpacity(0.25),
                          AppColors.accentPrimary.withOpacity(0.05),
                        ],
                      ),
                      border: Border.all(
                        color: AppColors.accentPrimary.withOpacity(0.4),
                        width: 1.5,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.accentPrimary.withOpacity(0.3),
                          blurRadius: 30,
                          spreadRadius: 5,
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.psychology_rounded,
                      size: 48,
                      color: AppColors.accentPrimary,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 40),
        const Text(
          'Analysis Running',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 24,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.5,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Attention U-Net++ processing your MRI data',
          style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
        ),
        const SizedBox(height: 32),
        // Step list
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            children: _steps.asMap().entries.map((e) {
              final isDone = e.key < _currentStep;
              final isActive = e.key == _currentStep;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: 22,
                      height: 22,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isDone
                            ? AppColors.success.withOpacity(0.15)
                            : isActive
                                ? AppColors.accentPrimary.withOpacity(0.15)
                                : Colors.transparent,
                        border: Border.all(
                          color: isDone
                              ? AppColors.success
                              : isActive
                                  ? AppColors.accentPrimary
                                  : AppColors.textMuted,
                          width: 1.5,
                        ),
                      ),
                      child: Center(
                        child: isDone
                            ? const Icon(Icons.check_rounded, size: 12, color: AppColors.success)
                            : isActive
                                ? Container(
                                    width: 6,
                                    height: 6,
                                    decoration: const BoxDecoration(
                                      color: AppColors.accentPrimary,
                                      shape: BoxShape.circle,
                                    ),
                                  )
                                : null,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      e.value,
                      style: TextStyle(
                        color: isDone
                            ? AppColors.success
                            : isActive
                                ? AppColors.textPrimary
                                : AppColors.textMuted,
                        fontSize: 13,
                        fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildSuccessState() {
    return Column(
      children: [
        Container(
          width: 100,
          height: 100,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.success.withOpacity(0.1),
            border: Border.all(color: AppColors.success.withOpacity(0.4), width: 2),
            boxShadow: [
              BoxShadow(color: AppColors.success.withOpacity(0.3), blurRadius: 30, spreadRadius: 5),
            ],
          ),
          child: const Icon(Icons.check_rounded, size: 48, color: AppColors.success),
        ),
        const SizedBox(height: 24),
        const Text(
          'Analysis Complete',
          style: TextStyle(color: AppColors.success, fontSize: 24, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 8),
        Text(
          'Loading your results...',
          style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
        ),
      ],
    );
  }

  Widget _buildErrorState() {
    return Column(
      children: [
        Container(
          width: 100,
          height: 100,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.danger.withOpacity(0.1),
            border: Border.all(color: AppColors.danger.withOpacity(0.4), width: 2),
          ),
          child: const Icon(Icons.error_outline_rounded, size: 48, color: AppColors.danger),
        ),
        const SizedBox(height: 24),
        const Text(
          'Analysis Failed',
          style: TextStyle(color: AppColors.danger, fontSize: 24, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 12),
        Text(
          _errorMessage!,
          style: const TextStyle(color: AppColors.textSecondary, fontSize: 13),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 28),
        GestureDetector(
          onTap: () => Navigator.pop(context),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
            decoration: BoxDecoration(
              color: AppColors.bgCard,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.border),
            ),
            child: const Text(
              'Go Back',
              style: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
            ),
          ),
        ),
      ],
    );
  }
}
