
// Tumor Result Model - Save as lib/models/tumor_result.dart
class TumorResult {
  final Map<String, double> confidenceScores;
  final Map<String, double> diceScores;
  final Map<String, double> iouScores;
  final String clinicalAssessment;
  final String riskLevel;
  final Map<String, dynamic> metrics;
  final Map<String, String> probImagesB64;
  final DateTime analysisTime;

  TumorResult({
    required this.confidenceScores,
    required this.diceScores,
    required this.iouScores,
    required this.clinicalAssessment,
    required this.riskLevel,
    required this.metrics,
    required this.probImagesB64,
    required this.analysisTime,
  });

  factory TumorResult.fromJson(Map<String, dynamic> json) {
    return TumorResult(
      confidenceScores: Map<String, double>.from(json['confidence_scores'] ?? {}),
      diceScores: Map<String, double>.from(json['dice_scores'] ?? {}),
      iouScores: Map<String, double>.from(json['iou_scores'] ?? {}),
      clinicalAssessment: json['clinical_assessment'] ?? 'Assessment not available',
      riskLevel: json['risk_level'] ?? 'Unknown',
      metrics: json['metrics'] ?? {},
      probImagesB64: Map<String, String>.from(json['probability_images'] ?? {}),
      analysisTime: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
    );
  }

  // Getters for easy access
  double get overallConfidence => confidenceScores['overall'] ?? 0.0;
  double get enhancingConfidence => confidenceScores['Enhancing Tumor'] ?? 0.0;
  double get coreConfidence => confidenceScores['Tumor Core'] ?? 0.0;
  double get wholeConfidence => confidenceScores['Whole Tumor'] ?? 0.0;

  bool get tumorDetected => overallConfidence > 0.3;

  String get confidenceLevelText {
    if (overallConfidence > 0.7) return 'Very High';
    if (overallConfidence > 0.5) return 'High';
    if (overallConfidence > 0.3) return 'Moderate';
    return 'Low';
  }

  // Get highest confidence tumor type
  String get dominantTumorType {
    double maxConf = 0.0;
    String dominantType = 'None';

    confidenceScores.forEach((type, confidence) {
      if (type != 'overall' && confidence > maxConf) {
        maxConf = confidence;
        dominantType = type;
      }
    });

    return dominantType;
  }
}
