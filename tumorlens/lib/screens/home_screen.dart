
// ignore_for_file: prefer_const_literals_to_create_immutables, prefer_const_constructors

// Main Home Screen - Save as lib/screens/home_screen.dart
import 'package:brain_tumor_detection/screens/upload_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';


class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Brain Tumor Detection',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: Padding(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Welcome Header
            Container(
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.blue[400]!, Colors.blue[600]!],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(15),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.medical_information, 
                       color: Colors.white, size: 40),
                  SizedBox(height: 10),
                  Text(
                    'AI-Powered Brain Tumor Analysis',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Advanced deep learning for medical diagnosis',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 30),

            // Features Grid
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                crossAxisSpacing: 15,
                mainAxisSpacing: 15,
                children: [
                  _buildFeatureCard(
                    context,
                    'New Analysis',
                    'Upload MRI scans for tumor detection',
                    Icons.upload_file,
                    Colors.green,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => UploadScreen()),
                    ),
                  ),
                  _buildFeatureCard(
                    context,
                    'Model Info',
                    'View AI model specifications',
                    Icons.info_outline,
                    Colors.blue,
                    () => _showModelInfo(context),
                  ),
                  _buildFeatureCard(
                    context,
                    'History',
                    'View previous analyses',
                    Icons.history,
                    Colors.orange,
                    () => _showHistory(context),
                  ),
                  _buildFeatureCard(
                    context,
                    'Settings',
                    'Configure app preferences',
                    Icons.settings,
                    Colors.purple,
                    () => _showSettings(context),
                  ),
                ],
              ),
            ),

            SizedBox(height: 20),

            // Model Status
            Consumer<AIService>(
              builder: (context, aiService, child) {
                return Container(
                  padding: EdgeInsets.all(15),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.circle,
                        color: Colors.green,
                        size: 16,
                      ),
                      SizedBox(width: 10),
                      Text('AI Model: Ready for Analysis'),
                      Spacer(),
                      Text('v1.0'),
                    ],
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureCard(
    BuildContext context,
    String title,
    String subtitle,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Card(
      elevation: 4,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: color, size: 40),
              SizedBox(height: 10),
              Text(
                title,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 5),
              Text(
                subtitle,
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showModelInfo(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('AI Model Information'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildInfoRow('Architecture', 'U-Net Deep Learning'),
            _buildInfoRow('Parameters', '7.7M'),
            _buildInfoRow('Training Data', 'BraTS2020'),
            _buildInfoRow('Input Modalities', 'FLAIR, T1, T1CE, T2'),
            _buildInfoRow('Performance', '50.9% avg confidence'),
            _buildInfoRow('Robustness', '±1.1% across degradations'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label: ', style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  void _showHistory(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Analysis history - Coming soon!')),
    );
  }

  void _showSettings(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Settings - Coming soon!')),
    );
  }
}
