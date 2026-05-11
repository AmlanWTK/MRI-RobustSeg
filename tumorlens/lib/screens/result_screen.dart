
// Results Screen - Save as lib/screens/results_screen.dart
import 'package:brain_tumor_detection/widgets/confidence_meter.dart';
import 'package:flutter/material.dart';
import '../models/tumor_result.dart';


class ResultsScreen extends StatelessWidget {
  final TumorResult result;

  const ResultsScreen({Key? key, required this.result}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Analysis Results'),
        actions: [
          IconButton(
            icon: Icon(Icons.share),
            onPressed: () => _shareResults(context),
          ),
          IconButton(
            icon: Icon(Icons.picture_as_pdf),
            onPressed: () => _generateReport(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Clinical Assessment Card
            _buildAssessmentCard(),
            SizedBox(height: 16),

            // Overall Confidence
            _buildConfidenceCard(),
            SizedBox(height: 16),

            // Detailed Metrics
            _buildMetricsCard(),
            SizedBox(height: 16),

            // Tumor Type Analysis
            _buildTumorAnalysisCard(),
            SizedBox(height: 16),

            // Technical Details
            _buildTechnicalDetailsCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildAssessmentCard() {
    Color assessmentColor = result.tumorDetected ? Colors.red : Colors.green;
    IconData assessmentIcon = result.tumorDetected ? Icons.warning : Icons.check_circle;

    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(assessmentIcon, color: assessmentColor, size: 28),
                SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Clinical Assessment',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
            SizedBox(height: 12),
            Text(
              result.clinicalAssessment,
              style: TextStyle(
                fontSize: 16,
                color: assessmentColor,
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Confidence Level: ${result.confidenceLevelText}',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
            ),
            Text(
              'Risk Level: ${result.riskLevel}',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConfidenceCard() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Overall Confidence',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            ConfidenceMeter(
              confidence: result.overallConfidence,
              title: 'Tumor Detection',
              subtitle: '${(result.overallConfidence * 100).toStringAsFixed(1)}%',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricsCard() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Performance Metrics',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            _buildMetricRow('Dice Scores', result.diceScores),
            SizedBox(height: 12),
            _buildMetricRow('IoU Scores', result.iouScores),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricRow(String title, Map<String, double> scores) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        SizedBox(height: 8),
        ...scores.entries.map((entry) => Padding(
          padding: EdgeInsets.only(bottom: 4),
          child: Row(
            children: [
              Text('${entry.key}: '),
              Text(
                '${(entry.value * 100).toStringAsFixed(1)}%',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
        )).toList(),
      ],
    );
  }

  Widget _buildTumorAnalysisCard() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Tumor Classification',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            _buildTumorTypeRow('Enhancing Tumor', result.enhancingConfidence, Colors.red),
            _buildTumorTypeRow('Tumor Core', result.coreConfidence, Colors.blue),
            _buildTumorTypeRow('Whole Tumor', result.wholeConfidence, Colors.green),
            SizedBox(height: 12),
            Text(
              'Dominant Type: ${result.dominantTumorType}',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTumorTypeRow(String type, double confidence, Color color) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            width: 16,
            height: 16,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Text(type, style: TextStyle(fontSize: 16)),
          ),
          Text(
            '${(confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTechnicalDetailsCard() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Technical Details',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            _buildDetailRow('Analysis Time', result.analysisTime.toString().split('.')[0]),
            _buildDetailRow('Model Type', 'U-Net Deep Learning'),
            _buildDetailRow('Processing Time', result.metrics['processing_time']?.toString() ?? 'N/A'),
            _buildDetailRow('Model Version', result.metrics['model_version']?.toString() ?? 'N/A'),
            _buildDetailRow('Modalities Processed', result.metrics['modalities_processed']?.toString() ?? 'N/A'),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text('$label: ', style: TextStyle(fontWeight: FontWeight.w600)),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  void _shareResults(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Sharing results...')),
    );
  }

  void _generateReport(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Generating PDF report...')),
    );
  }
}
