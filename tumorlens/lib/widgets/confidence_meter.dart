
// Confidence Meter Widget - Save as lib/widgets/confidence_meter.dart
import 'package:flutter/material.dart';

class ConfidenceMeter extends StatelessWidget {
  final double confidence;
  final String title;
  final String subtitle;

  const ConfidenceMeter({
    Key? key,
    required this.confidence,
    required this.title,
    required this.subtitle,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Color confidenceColor = _getConfidenceColor(confidence);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              title,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            Text(
              subtitle,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: confidenceColor,
              ),
            ),
          ],
        ),
        SizedBox(height: 8),
        LinearProgressIndicator(
          value: confidence,
          backgroundColor: Colors.grey[300],
          valueColor: AlwaysStoppedAnimation<Color>(confidenceColor),
          minHeight: 8,
        ),
        SizedBox(height: 4),
        Text(
          _getConfidenceText(confidence),
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.7) return Colors.red;
    if (confidence >= 0.5) return Colors.orange;
    if (confidence >= 0.3) return Colors.yellow[700]!;
    return Colors.green;
  }

  String _getConfidenceText(double confidence) {
    if (confidence >= 0.7) return 'Very High Confidence';
    if (confidence >= 0.5) return 'High Confidence';
    if (confidence >= 0.3) return 'Moderate Confidence';
    return 'Low Confidence';
  }
}
