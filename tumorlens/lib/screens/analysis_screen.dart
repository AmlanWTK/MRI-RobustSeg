
// Analysis Screen - Save as lib/screens/analysis_screen.dart
import 'package:brain_tumor_detection/models/tumor_result.dart';
import 'package:brain_tumor_detection/screens/result_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:typed_data';


class AnalysisScreen extends StatefulWidget {
  final List<Uint8List> imageBytes;

  const AnalysisScreen({Key? key, required this.imageBytes}) : super(key: key);

  @override
  _AnalysisScreenState createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;
  bool _analysisComplete = false;
  TumorResult? _result;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: Duration(seconds: 2),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.repeat();

    _startAnalysis();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _startAnalysis() async {
    try {
      final aiService = Provider.of<AIService>(context, listen: false);

      // Use simulation for now - replace with real analysis when backend is ready
      final result = await aiService.simulateAnalysis();
      // final result = await aiService.analyzeMRI(widget.imageBytes); // Use this for real backend

      setState(() {
        _result = result;
        _analysisComplete = true;
        _animationController.stop();
      });

      // Navigate to results after a short delay
      await Future.delayed(Duration(seconds: 1));
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(result: result),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _analysisComplete = true;
        _animationController.stop();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AI Analysis'),
        automaticallyImplyLeading: false, // Prevent back navigation during analysis
      ),
      body: Padding(
        padding: EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Brain Icon with Animation
            AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return Transform.scale(
                  scale: 0.8 + (_animation.value * 0.2),
                  child: Container(
                    padding: EdgeInsets.all(30),
                    decoration: BoxDecoration(
                      color: Colors.blue.withOpacity(0.1),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.psychology,
                      size: 80,
                      color: Colors.blue,
                    ),
                  ),
                );
              },
            ),

            SizedBox(height: 40),

            // Title
            Text(
              'AI Brain Analysis',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800],
              ),
            ),

            SizedBox(height: 20),

            // Status or Error Message
            if (_errorMessage != null)
              _buildErrorWidget()
            else if (_analysisComplete && _result != null)
              _buildSuccessWidget()
            else
              _buildProgressWidget(),

            SizedBox(height: 40),

            // Progress Indicator
            if (!_analysisComplete)
              CircularProgressIndicator(
                strokeWidth: 3,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
              ),

            SizedBox(height: 20),

            // Cancel Button (only show during analysis)
            if (!_analysisComplete)
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: Text('Cancel Analysis'),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressWidget() {
    return Consumer<AIService>(
      builder: (context, aiService, child) {
        return Column(
          children: [
            Text(
              aiService.analysisProgress.isEmpty 
                ? 'Initializing AI model...' 
                : aiService.analysisProgress,
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 20),
            Text(
              'Please wait while we analyze your MRI scans',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[500],
              ),
              textAlign: TextAlign.center,
            ),
          ],
        );
      },
    );
  }

  Widget _buildSuccessWidget() {
    return Column(
      children: [
        Icon(
          Icons.check_circle,
          color: Colors.green,
          size: 48,
        ),
        SizedBox(height: 16),
        Text(
          'Analysis Complete!',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.green,
          ),
        ),
        SizedBox(height: 8),
        Text(
          'Redirecting to results...',
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  Widget _buildErrorWidget() {
    return Column(
      children: [
        Icon(
          Icons.error_outline,
          color: Colors.red,
          size: 48,
        ),
        SizedBox(height: 16),
        Text(
          'Analysis Failed',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.red,
          ),
        ),
        SizedBox(height: 8),
        Text(
          _errorMessage!,
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey[600],
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 20),
        ElevatedButton(
          onPressed: () {
            Navigator.pop(context);
          },
          child: Text('Go Back'),
        ),
      ],
    );
  }
}
