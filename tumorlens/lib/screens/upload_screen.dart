
// ignore_for_file: prefer_const_constructors

// Upload Screen - Save as lib/screens/upload_screen.dart
import 'package:brain_tumor_detection/screens/analysis_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import 'dart:typed_data';


class UploadScreen extends StatefulWidget {
  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  Map<String, File?> uploadedImages = {
    'FLAIR': null,
    'T1': null,
    'T1CE': null,
    'T2': null,
  };

  final ImagePicker _picker = ImagePicker();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload MRI Scans'),
        actions: [
          IconButton(
            icon: Icon(Icons.help_outline),
            onPressed: () => _showHelpDialog(),
          ),
        ],
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Instructions
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.blue[200]!),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info, color: Colors.blue),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Upload all 4 MRI modalities for optimal analysis. Missing modalities will be handled automatically.',
                      style: TextStyle(color: Colors.blue[800]),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 20),

            // Upload Cards
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                crossAxisSpacing: 10,
                mainAxisSpacing: 10,
                children: uploadedImages.keys.map((modality) {
                  return _buildUploadCard(modality);
                }).toList(),
              ),
            ),

            SizedBox(height: 20),

            // Progress Indicator
            Consumer<AIService>(
              builder: (context, aiService, child) {
                if (aiService.isAnalyzing) {
                  return Column(
                    children: [
                      LinearProgressIndicator(),
                      SizedBox(height: 10),
                      Text(aiService.analysisProgress),
                    ],
                  );
                }
                return SizedBox.shrink();
              },
            ),

            SizedBox(height: 20),

            // Analyze Button
            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton(
                onPressed: _canAnalyze() ? _analyzeImages : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: Text(
                  'Analyze Brain Tumor',
                  style: TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploadCard(String modality) {
    File? image = uploadedImages[modality];
    bool hasImage = image != null;

    return Card(
      elevation: 4,
      child: InkWell(
        onTap: () => _pickImage(modality),
        child: Padding(
          padding: EdgeInsets.all(12),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (hasImage)
                Expanded(
                  child: Image.file(
                    image,
                    fit: BoxFit.cover,
                    width: double.infinity,
                  ),
                )
              else
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.add_photo_alternate,
                        size: 40,
                        color: Colors.grey,
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Tap to upload',
                        style: TextStyle(color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              SizedBox(height: 8),
              Container(
                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: hasImage ? Colors.green : Colors.grey[300],
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  modality,
                  style: TextStyle(
                    color: hasImage ? Colors.white : Colors.grey[600],
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _pickImage(String modality) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 100,
      );

      if (image != null) {
        setState(() {
          uploadedImages[modality] = File(image.path);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error picking image: $e')),
      );
    }
  }

  bool _canAnalyze() {
    return uploadedImages.values.any((image) => image != null);
  }

  Future<void> _analyzeImages() async {
    try {
      // Convert images to bytes
      List<Uint8List> imageBytes = [];
      for (String modality in ['FLAIR', 'T1', 'T1CE', 'T2']) {
        if (uploadedImages[modality] != null) {
          imageBytes.add(await uploadedImages[modality]!.readAsBytes());
        } else {
          // Create empty placeholder
          imageBytes.add(Uint8List(0));
        }
      }

      // Navigate to analysis screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => AnalysisScreen(imageBytes: imageBytes),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error starting analysis: $e')),
      );
    }
  }

  void _showHelpDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('MRI Modalities Guide'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildModalityInfo('FLAIR', 'Fluid Attenuated Inversion Recovery'),
              _buildModalityInfo('T1', 'T1-weighted imaging'),
              _buildModalityInfo('T1CE', 'T1-weighted with contrast enhancement'),
              _buildModalityInfo('T2', 'T2-weighted imaging'),
              SizedBox(height: 16),
              Text(
                'Note: The AI model works best with all 4 modalities, but can analyze with fewer images available.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Got it'),
          ),
        ],
      ),
    );
  }

  Widget _buildModalityInfo(String modality, String description) {
    return Padding(
      padding: EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            modality,
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          Text(
            description,
            style: TextStyle(color: Colors.grey[600], fontSize: 12),
          ),
        ],
      ),
    );
  }
}
