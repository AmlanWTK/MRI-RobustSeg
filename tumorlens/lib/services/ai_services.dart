
// AI Service - Save as lib/services/ai_service.dart
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import '../models/tumor_result.dart';

class AIService extends ChangeNotifier {
  static const String baseUrl = 'https://puny-ads-wear.loca.lt/api';  // Using localtunnel for physical device testing

  bool _isAnalyzing = false;
  String _analysisProgress = '';
  TumorResult? _lastResult;

  bool get isAnalyzing => _isAnalyzing;
  String get analysisProgress => _analysisProgress;
  TumorResult? get lastResult => _lastResult;




  Future<TumorResult> analyzeMRI(List<Uint8List> mriImages) async {
    _isAnalyzing = true;
    _analysisProgress = 'Uploading images...';
    notifyListeners();

    try {
      print("🌐 Sending analysis request to: $baseUrl/analyze");
      // Create multipart request
      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/analyze'));
      request.headers['Bypass-Tunnel-Reminder'] = 'true';

      // Add MRI modality images
      List<String> modalities = ['flair', 't1', 't1ce', 't2'];
      for (int i = 0; i < mriImages.length && i < 4; i++) {
        if (mriImages[i].isNotEmpty) {
          request.files.add(
            http.MultipartFile.fromBytes(
              modalities[i],
              mriImages[i],
              filename: '${modalities[i]}.png',
            ),
          );
        }
      }

      _analysisProgress = 'Processing with AI model...';
      notifyListeners();

      // Send request
      print("⏳ Waiting for backend response...");
      var response = await request.send();

      if (response.statusCode == 200) {
        print("✅ Backend response received successfully (200 OK)");
        _analysisProgress = 'Generating results...';
        notifyListeners();

        String responseBody = await response.stream.bytesToString();
        Map<String, dynamic> json = jsonDecode(responseBody);

        _lastResult = TumorResult.fromJson(json);
        _isAnalyzing = false;
        _analysisProgress = '';
        notifyListeners();

        return _lastResult!;
      } else {
        print("❌ Backend returned status code: ${response.statusCode}");
        throw Exception('Analysis failed: ${response.statusCode}');
      }
    } catch (e, stack) {
      print("❌ EXCEPTION IN ANALYZEMRI: $e");
      print("Stacktrace: $stack");
      _isAnalyzing = false;
      _analysisProgress = '';
      notifyListeners();
      throw Exception('Error during analysis: $e');
  }}

  // Simulate analysis for testing without backend
  Future<TumorResult> simulateAnalysis() async {
    _isAnalyzing = true;
    _analysisProgress = 'Initializing AI model...';
    notifyListeners();

    await Future.delayed(Duration(seconds: 1));

    _analysisProgress = 'Processing MRI images...';
    notifyListeners();

    await Future.delayed(Duration(seconds: 2));

    _analysisProgress = 'Analyzing tumor patterns...';
    notifyListeners();

    await Future.delayed(Duration(seconds: 2));

    _analysisProgress = 'Generating results...';
    notifyListeners();

    await Future.delayed(Duration(seconds: 1));

    // Create mock result
    Map<String, dynamic> mockData = {
      'confidence_scores': {
        'Enhancing Tumor': 0.765,
        'Tumor Core': 0.607,
        'Whole Tumor': 0.472,
        'overall': 0.615
      },
      'dice_scores': {
        'Enhancing Tumor': 0.823,
        'Tumor Core': 0.745,
        'Whole Tumor': 0.681
      },
      'iou_scores': {
        'Enhancing Tumor': 0.701,
        'Tumor Core': 0.593,
        'Whole Tumor': 0.516
      },
      'clinical_assessment': 'TUMOR DETECTED - Recommend clinical review',
      'risk_level': 'High',
      'timestamp': DateTime.now().toIso8601String(),
      'metrics': {
        'processing_time': '5.8 seconds',
        'model_version': '1.0',
        'modalities_processed': 4
      }
    };

    _lastResult = TumorResult.fromJson(mockData);
    _isAnalyzing = false;
    _analysisProgress = '';
    notifyListeners();

    return _lastResult!;
  }
}
