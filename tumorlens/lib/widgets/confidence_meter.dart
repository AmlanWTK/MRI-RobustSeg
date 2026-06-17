// ignore_for_file: prefer_const_constructors

import 'package:brain_tumor_detection/main.dart';
import 'package:flutter/material.dart';


// ─── Confidence Meter Widget ──────────────────────────────────────────────────
class ConfidenceMeter extends StatelessWidget {
  final double confidence;
  final String title;
  final String subtitle;
  final Color? color;

  const ConfidenceMeter({
    Key? key,
    required this.confidence,
    required this.title,
    required this.subtitle,
    this.color,
  }) : super(key: key);

  Color get _barColor => color ?? _defaultColor;

  Color get _defaultColor {
    if (confidence >= 0.7) return AppColors.danger;
    if (confidence >= 0.5) return AppColors.warning;
    if (confidence >= 0.3) return AppColors.warning;
    return AppColors.success;
  }

  String get _levelText {
    if (confidence >= 0.7) return 'Very High';
    if (confidence >= 0.5) return 'High';
    if (confidence >= 0.3) return 'Moderate';
    return 'Low';
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            // Percentage badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: _barColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                subtitle,
                style: TextStyle(
                  color: _barColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        // Track
        Stack(
          children: [
            Container(
              height: 6,
              width: double.infinity,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            FractionallySizedBox(
              widthFactor: confidence.clamp(0.0, 1.0),
              child: Container(
                height: 6,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(3),
                  gradient: LinearGradient(
                    colors: [_barColor.withOpacity(0.7), _barColor],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: _barColor.withOpacity(0.4),
                      blurRadius: 6,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          '$_levelText confidence',
          style: const TextStyle(color: AppColors.textMuted, fontSize: 11),
        ),
      ],
    );
  }
}
