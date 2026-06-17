// ignore_for_file: prefer_const_constructors

import 'dart:convert';
import 'package:brain_tumor_detection/main.dart';
import 'package:brain_tumor_detection/widgets/confidence_meter.dart';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/tumor_result.dart';


// ─── Results Screen ───────────────────────────────────────────────────────────
class ResultsScreen extends StatefulWidget {
  final TumorResult result;

  const ResultsScreen({Key? key, required this.result}) : super(key: key);

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> with SingleTickerProviderStateMixin {
  late AnimationController _fadeController;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..forward();
    _fadeAnim = CurvedAnimation(parent: _fadeController, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: Column(
            children: [
              _buildTopBar(context),
              Expanded(
                child: SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    children: [
                      const SizedBox(height: 8),
                      _buildAssessmentBanner(),
                      const SizedBox(height: 20),
                      _buildMetricsRow(),
                      const SizedBox(height: 24),
                      if (widget.result.probImagesB64.isNotEmpty) ...[
                        _buildSectionLabel('AI SEGMENTATION MASKS'),
                        const SizedBox(height: 16),
                        _buildSegmentationImages(),
                        const SizedBox(height: 24),
                      ],
                      _buildSectionLabel('SEGMENTATION CONFIDENCE'),
                      const SizedBox(height: 16),
                      _buildConfidencePanel(),
                      const SizedBox(height: 20),
                      _buildSectionLabel('DICE & IoU SCORES'),
                      const SizedBox(height: 16),
                      _buildBarChartPanel(),
                      const SizedBox(height: 20),
                      _buildSectionLabel('TUMOR CLASSIFICATION'),
                      const SizedBox(height: 16),
                      _buildTumorClassification(),
                      const SizedBox(height: 20),
                      _buildSectionLabel('CLINICAL RECOMMENDATIONS'),
                      const SizedBox(height: 16),
                      _buildRecommendations(),
                      const SizedBox(height: 20),
                      _buildSectionLabel('TECHNICAL DETAILS'),
                      const SizedBox(height: 16),
                      _buildTechnicalDetails(),
                      const SizedBox(height: 32),
                      _buildActionButtons(context),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        children: [
          GestureDetector(
            onTap: () => Navigator.popUntil(context, (r) => r.isFirst),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.border),
              ),
              child: const Icon(Icons.home_rounded, color: AppColors.textSecondary, size: 20),
            ),
          ),
          const SizedBox(width: 16),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Analysis Report',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                'AI Segmentation Results',
                style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
              ),
            ],
          ),
          const Spacer(),
          GestureDetector(
            onTap: () => _shareResults(context),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.border),
              ),
              child: const Icon(Icons.ios_share_rounded, color: AppColors.textSecondary, size: 20),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAssessmentBanner() {
    final detected = widget.result.tumorDetected;
    final isHigh = widget.result.riskLevel == 'High';
    final bannerColor = isHigh
        ? AppColors.danger
        : detected
            ? AppColors.warning
            : AppColors.success;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: bannerColor.withOpacity(0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: bannerColor.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: bannerColor.withOpacity(0.15),
            ),
            child: Icon(
              isHigh
                  ? Icons.warning_amber_rounded
                  : detected
                      ? Icons.info_rounded
                      : Icons.check_circle_rounded,
              color: bannerColor,
              size: 26,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.result.clinicalAssessment,
                  style: TextStyle(
                    color: bannerColor,
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    height: 1.3,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Risk Level: ${widget.result.riskLevel}  ·  Confidence: ${widget.result.confidenceLevelText}',
                  style: TextStyle(
                    color: bannerColor.withOpacity(0.7),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsRow() {
    final conf = widget.result.overallConfidence;
    final avgDice = widget.result.diceScores.values
        .fold<double>(0.0, (double a, double b) => a + b) /
        widget.result.diceScores.length;
    final avgIou = widget.result.iouScores.values
        .fold<double>(0.0, (double a, double b) => a + b) /
        widget.result.iouScores.length;

    return Row(
      children: [
        _buildMetricTile('${(conf * 100).toStringAsFixed(1)}%', 'Confidence', AppColors.accentPrimary),
        const SizedBox(width: 12),
        _buildMetricTile('${(avgDice * 100).toStringAsFixed(1)}%', 'Avg Dice', AppColors.info),
        const SizedBox(width: 12),
        _buildMetricTile('${(avgIou * 100).toStringAsFixed(1)}%', 'Avg IoU', AppColors.whole),
      ],
    );
  }

  Color _getColorForClass(String className) {
    if (className == 'Enhancing Tumor') return AppColors.enhancing;
    if (className == 'Tumor Core') return AppColors.core;
    if (className == 'Whole Tumor') return AppColors.whole;
    return AppColors.accentPrimary;
  }

  Widget _buildSegmentationImages() {
    final images = widget.result.probImagesB64;
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      physics: const BouncingScrollPhysics(),
      child: Row(
        children: images.entries.map((entry) {
          final color = _getColorForClass(entry.key);
          return Container(
            margin: const EdgeInsets.only(right: 16),
            width: 160,
            decoration: BoxDecoration(
              color: AppColors.bgCard,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                ClipRRect(
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
                  child: Image.memory(
                    base64Decode(entry.value),
                    height: 160,
                    width: 160,
                    fit: BoxFit.cover,
                    color: color.withOpacity(0.8), // Tint the grayscale mask
                    colorBlendMode: BlendMode.srcIn,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  alignment: Alignment.center,
                  child: Text(
                    entry.key,
                    style: TextStyle(
                      color: color,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildMetricTile(String value, String label, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 18),
        decoration: BoxDecoration(
          color: color.withOpacity(0.07),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.25)),
        ),
        child: Column(
          children: [
            Text(
              value,
              style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: const TextStyle(color: AppColors.textSecondary, fontSize: 11),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionLabel(String label) {
    return Row(
      children: [
        Container(
          width: 3,
          height: 14,
          decoration: BoxDecoration(
            color: AppColors.accentPrimary,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 10),
        Text(
          label,
          style: const TextStyle(
            color: AppColors.textMuted,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 2.5,
          ),
        ),
      ],
    );
  }

  Widget _buildConfidencePanel() {
    final scores = [
      _ScoreData('Enhancing Tumor', widget.result.enhancingConfidence, AppColors.enhancing),
      _ScoreData('Tumor Core', widget.result.coreConfidence, AppColors.core),
      _ScoreData('Whole Tumor', widget.result.wholeConfidence, AppColors.whole),
      _ScoreData('Overall', widget.result.overallConfidence, AppColors.accentPrimary),
    ];

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: scores.map((s) => Padding(
          padding: const EdgeInsets.only(bottom: 16),
          child: ConfidenceMeter(
            confidence: s.value,
            title: s.label,
            subtitle: '${(s.value * 100).toStringAsFixed(1)}%',
            color: s.color,
          ),
        )).toList(),
      ),
    );
  }

  Widget _buildBarChartPanel() {
    final diceData = widget.result.diceScores;
    final iouData = widget.result.iouScores;
    final keys = ['Enhancing\nTumor', 'Tumor\nCore', 'Whole\nTumor'];
    final diceKeys = ['Enhancing Tumor', 'Tumor Core', 'Whole Tumor'];

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          // Legend
          Row(
            children: [
              _legendDot(AppColors.accentPrimary, 'Dice Score'),
              const SizedBox(width: 20),
              _legendDot(AppColors.whole, 'IoU Score'),
            ],
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 160,
            child: BarChart(
              BarChartData(
                alignment: BarChartAlignment.spaceAround,
                maxY: 1.0,
                barTouchData: BarTouchData(
                  touchTooltipData: BarTouchTooltipData(
                    tooltipBgColor: AppColors.bgSurface,
                    getTooltipItem: (group, groupIndex, rod, rodIndex) {
                      return BarTooltipItem(
                        '${(rod.toY * 100).toStringAsFixed(1)}%',
                        const TextStyle(color: AppColors.textPrimary, fontSize: 12, fontWeight: FontWeight.w600),
                      );
                    },
                  ),
                ),
                titlesData: FlTitlesData(
                  show: true,
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 36,
                      getTitlesWidget: (value, _) => Text(
                        '${(value * 100).toInt()}%',
                        style: const TextStyle(color: AppColors.textMuted, fontSize: 10),
                      ),
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 40,
                      getTitlesWidget: (value, _) => Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          keys[value.toInt()],
                          style: const TextStyle(color: AppColors.textSecondary, fontSize: 10),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),
                  ),
                  topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: 0.25,
                  getDrawingHorizontalLine: (_) => FlLine(color: AppColors.border, strokeWidth: 1),
                ),
                borderData: FlBorderData(show: false),
                barGroups: List.generate(3, (i) => BarChartGroupData(
                  x: i,
                  barRods: [
                    BarChartRodData(
                      toY: diceData[diceKeys[i]] ?? 0,
                      color: AppColors.accentPrimary,
                      width: 14,
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
                    ),
                    BarChartRodData(
                      toY: iouData[diceKeys[i]] ?? 0,
                      color: AppColors.whole,
                      width: 14,
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
                    ),
                  ],
                  barsSpace: 4,
                )),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(3)),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(color: AppColors.textSecondary, fontSize: 12)),
      ],
    );
  }

  Widget _buildTumorClassification() {
    final types = [
      _TumorType('Enhancing Tumor', widget.result.enhancingConfidence, AppColors.enhancing,
          'Active tumor cells with blood-brain barrier breakdown'),
      _TumorType('Tumor Core', widget.result.coreConfidence, AppColors.core,
          'Necrotic and solid tumor region'),
      _TumorType('Whole Tumor', widget.result.wholeConfidence, AppColors.whole,
          'Complete tumor extent including edema'),
    ];

    return Column(
      children: types.map((t) => Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.bgCard,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: widget.result.dominantTumorType == t.name
                ? t.color.withOpacity(0.5)
                : AppColors.border,
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: t.color.withOpacity(0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: Text(
                  '${(t.confidence * 100).toStringAsFixed(0)}%',
                  style: TextStyle(
                    color: t.color,
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        t.name,
                        style: const TextStyle(
                          color: AppColors.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      if (widget.result.dominantTumorType == t.name) ...[
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: t.color.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            'DOMINANT',
                            style: TextStyle(
                              color: t.color,
                              fontSize: 9,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 1,
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    t.description,
                    style: const TextStyle(color: AppColors.textSecondary, fontSize: 11, height: 1.3),
                  ),
                ],
              ),
            ),
          ],
        ),
      )).toList(),
    );
  }

  Widget _buildRecommendations() {
    final confidence = widget.result.overallConfidence;
    List<_Recommendation> recs;

    if (confidence > 0.7) {
      recs = [
        _Recommendation(Icons.emergency_rounded, 'Immediate clinical review recommended', AppColors.danger),
        _Recommendation(Icons.medical_services_rounded, 'Consider contrast-enhanced MRI', AppColors.warning),
        _Recommendation(Icons.groups_rounded, 'Multidisciplinary team consultation advised', AppColors.warning),
        _Recommendation(Icons.monitor_heart_rounded, 'Monitor for neurological changes', AppColors.info),
      ];
    } else if (confidence > 0.4) {
      recs = [
        _Recommendation(Icons.check_circle_rounded, 'Clinical correlation recommended', AppColors.warning),
        _Recommendation(Icons.schedule_rounded, 'Follow-up imaging in 3–6 months', AppColors.info),
        _Recommendation(Icons.history_edu_rounded, 'Review patient symptoms and history', AppColors.info),
      ];
    } else {
      recs = [
        _Recommendation(Icons.check_circle_rounded, 'No immediate action required', AppColors.success),
        _Recommendation(Icons.calendar_today_rounded, 'Routine clinical follow-up', AppColors.success),
        _Recommendation(Icons.person_rounded, 'Patient counseling on normal findings', AppColors.info),
      ];
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: recs.map((r) => Padding(
          padding: const EdgeInsets.only(bottom: 14),
          child: Row(
            children: [
              Icon(r.icon, color: r.color, size: 18),
              const SizedBox(width: 14),
              Expanded(
                child: Text(
                  r.text,
                  style: const TextStyle(color: AppColors.textPrimary, fontSize: 13, height: 1.3),
                ),
              ),
            ],
          ),
        )).toList(),
      ),
    );
  }

  Widget _buildTechnicalDetails() {
    final details = [
      ['Model', 'Attention U-Net++'],
      ['Training Data', 'BraTS2020'],
      ['Parameters', '7.7 M'],
      ['Processing Time', widget.result.metrics['processing_time']?.toString() ?? 'N/A'],
      ['Model Version', widget.result.metrics['model_version']?.toString() ?? 'N/A'],
      ['Modalities Processed', '${widget.result.metrics['modalities_processed'] ?? 'N/A'}'],
      ['Analysis Timestamp', widget.result.analysisTime.toLocal().toString().split('.')[0]],
    ];

    return Container(
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: details.asMap().entries.map((e) {
          final isLast = e.key == details.length - 1;
          return Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 13),
                child: Row(
                  children: [
                    Text(e.value[0], style: const TextStyle(color: AppColors.textSecondary, fontSize: 13)),
                    const Spacer(),
                    Text(
                      e.value[1],
                      style: const TextStyle(color: AppColors.textPrimary, fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
              if (!isLast) const Divider(height: 0, color: AppColors.border),
            ],
          );
        }).toList(),
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: GestureDetector(
            onTap: () => Navigator.popUntil(context, (r) => r.isFirst),
            child: Container(
              height: 54,
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.home_rounded, color: AppColors.textSecondary, size: 20),
                  SizedBox(width: 8),
                  Text('Home', style: TextStyle(color: AppColors.textSecondary, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          flex: 2,
          child: GestureDetector(
            onTap: () => _generateReport(context),
            child: Container(
              height: 54,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                gradient: const LinearGradient(colors: [AppColors.accentGlow, AppColors.accentSecondary]),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.accentPrimary.withOpacity(0.3),
                    blurRadius: 20,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.picture_as_pdf_rounded, color: AppColors.bg, size: 20),
                  SizedBox(width: 8),
                  Text('Generate Report', style: TextStyle(color: AppColors.bg, fontWeight: FontWeight.w800)),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  void _shareResults(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Preparing results for sharing...'),
        backgroundColor: AppColors.bgCard,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  void _generateReport(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Generating PDF report...'),
        backgroundColor: AppColors.bgCard,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }
}

// ─── Helper Data Classes ──────────────────────────────────────────────────────
class _ScoreData {
  final String label;
  final double value;
  final Color color;
  const _ScoreData(this.label, this.value, this.color);
}

class _TumorType {
  final String name, description;
  final double confidence;
  final Color color;
  const _TumorType(this.name, this.confidence, this.color, this.description);
}

class _Recommendation {
  final IconData icon;
  final String text;
  final Color color;
  const _Recommendation(this.icon, this.text, this.color);
}
