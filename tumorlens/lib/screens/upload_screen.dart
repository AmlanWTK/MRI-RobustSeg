// ignore_for_file: prefer_const_constructors

import 'dart:io';
import 'dart:typed_data';

import 'package:brain_tumor_detection/main.dart';
import 'package:brain_tumor_detection/screens/analysis_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';


// ─── Upload Screen ────────────────────────────────────────────────────────────
class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> with TickerProviderStateMixin {
  final Map<String, File?> _images = {
    'FLAIR': null,
    'T1': null,
    'T1CE': null,
    'T2': null,
  };

  final Map<String, String> _modalityDescriptions = {
    'FLAIR': 'Fluid Attenuated\nInversion Recovery',
    'T1': 'T1-weighted\nImaging',
    'T1CE': 'T1 with Contrast\nEnhancement',
    'T2': 'T2-weighted\nImaging',
  };

  final Map<String, Color> _modalityColors = {
    'FLAIR': AppColors.accentPrimary,
    'T1': AppColors.info,
    'T1CE': AppColors.enhancing,
    'T2': AppColors.whole,
  };

  final ImagePicker _picker = ImagePicker();
  late AnimationController _shakeController;

  @override
  void initState() {
    super.initState();
    _shakeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
  }

  @override
  void dispose() {
    _shakeController.dispose();
    super.dispose();
  }

  int get _uploadedCount => _images.values.where((v) => v != null).length;
  bool get _canAnalyze => _uploadedCount > 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(context),
            Expanded(
              child: SingleChildScrollView(
                physics: const BouncingScrollPhysics(),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 8),
                      _buildProgressHeader(),
                      const SizedBox(height: 24),
                      _buildInfoBanner(),
                      const SizedBox(height: 24),
                      _buildModalityGrid(),
                      const SizedBox(height: 24),
                      _buildAnalysisButton(),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ),
          ],
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
            onTap: () => Navigator.pop(context),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.border),
              ),
              child: const Icon(Icons.arrow_back_rounded, color: AppColors.textSecondary, size: 20),
            ),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Upload MRI Scans',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                'Select imaging modalities',
                style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
              ),
            ],
          ),
          const Spacer(),
          GestureDetector(
            onTap: _showHelpDialog,
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.border),
              ),
              child: const Icon(Icons.help_outline_rounded, color: AppColors.textSecondary, size: 20),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              '$_uploadedCount / 4',
              style: const TextStyle(
                color: AppColors.accentPrimary,
                fontSize: 28,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(width: 8),
            const Text(
              'modalities uploaded',
              style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Stack(
          children: [
            Container(
              height: 4,
              width: double.infinity,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            AnimatedContainer(
              duration: const Duration(milliseconds: 400),
              curve: Curves.easeOut,
              height: 4,
              width: (MediaQuery.of(context).size.width - 40) * (_uploadedCount / 4),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(2),
                gradient: const LinearGradient(
                  colors: [AppColors.accentGlow, AppColors.accentSecondary],
                ),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.accentPrimary.withOpacity(0.5),
                    blurRadius: 6,
                  ),
                ],
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildInfoBanner() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.accentSoft.withOpacity(0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.accentPrimary.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline_rounded, color: AppColors.accentPrimary, size: 18),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Upload all 4 modalities for optimal analysis. Missing ones are handled automatically by the model.',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: 13,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModalityGrid() {
    final modalities = _images.keys.toList();
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 0.85,
      ),
      itemCount: modalities.length,
      itemBuilder: (_, i) => _ModalityCard(
        modality: modalities[i],
        description: _modalityDescriptions[modalities[i]]!,
        color: _modalityColors[modalities[i]]!,
        file: _images[modalities[i]],
        onTap: () => _pickImage(modalities[i]),
        onRemove: () => setState(() => _images[modalities[i]] = null),
      ),
    );
  }

  Widget _buildAnalysisButton() {
    return Consumer<AIService>(
      builder: (context, ai, _) {
        if (ai.isAnalyzing) {
          return _buildAnalyzingState(ai);
        }
        return GestureDetector(
          onTap: _canAnalyze ? _analyzeImages : null,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            height: 60,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              gradient: _canAnalyze
                  ? const LinearGradient(colors: [AppColors.accentGlow, AppColors.accentSecondary])
                  : null,
              color: _canAnalyze ? null : AppColors.bgCard,
              border: _canAnalyze ? null : Border.all(color: AppColors.border),
              boxShadow: _canAnalyze
                  ? [BoxShadow(color: AppColors.accentPrimary.withOpacity(0.35), blurRadius: 24, offset: const Offset(0, 8))]
                  : [],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.biotech_rounded,
                  color: _canAnalyze ? AppColors.bg : AppColors.textMuted,
                  size: 22,
                ),
                const SizedBox(width: 12),
                Text(
                  'Run Analysis',
                  style: TextStyle(
                    color: _canAnalyze ? AppColors.bg : AppColors.textMuted,
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildAnalyzingState(AIService ai) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.accentPrimary.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          LinearProgressIndicator(
            backgroundColor: AppColors.border,
            valueColor: const AlwaysStoppedAnimation<Color>(AppColors.accentPrimary),
            minHeight: 3,
          ),
          const SizedBox(height: 16),
          Text(
            ai.analysisProgress.isEmpty ? 'Initializing...' : ai.analysisProgress,
            style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImage(String modality) async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
      if (image != null) {
        setState(() => _images[modality] = File(image.path));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error picking image: $e')),
        );
      }
    }
  }

  Future<void> _analyzeImages() async {
    try {
      List<Uint8List> imageBytes = [];
      for (String modality in ['FLAIR', 'T1', 'T1CE', 'T2']) {
        if (_images[modality] != null) {
          imageBytes.add(await _images[modality]!.readAsBytes());
        } else {
          imageBytes.add(Uint8List(0));
        }
      }
      if (!mounted) return;
      Navigator.push(
        context,
        PageRouteBuilder(
          pageBuilder: (_, anim, __) => AnalysisScreen(imageBytes: imageBytes),
          transitionsBuilder: (_, anim, __, child) => FadeTransition(opacity: anim, child: child),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  void _showHelpDialog() {
    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: AppColors.bgCard,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: AppColors.border),
        ),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'MRI Modality Guide',
                style: TextStyle(color: AppColors.textPrimary, fontSize: 18, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 20),
              ..._modalityDescriptions.entries.map((e) => Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: _modalityColors[e.key]!.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        e.key,
                        style: TextStyle(
                          color: _modalityColors[e.key],
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        e.value.replaceAll('\n', ' '),
                        style: const TextStyle(color: AppColors.textSecondary, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              )),
              const SizedBox(height: 4),
              SizedBox(
                width: double.infinity,
                child: TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  style: TextButton.styleFrom(
                    backgroundColor: AppColors.accentPrimary.withOpacity(0.1),
                    foregroundColor: AppColors.accentPrimary,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  child: const Text('Got it', style: TextStyle(fontWeight: FontWeight.w700)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Modality Upload Card ─────────────────────────────────────────────────────
class _ModalityCard extends StatefulWidget {
  final String modality, description;
  final Color color;
  final File? file;
  final VoidCallback onTap, onRemove;

  const _ModalityCard({
    required this.modality,
    required this.description,
    required this.color,
    required this.file,
    required this.onTap,
    required this.onRemove,
  });

  @override
  State<_ModalityCard> createState() => _ModalityCardState();
}

class _ModalityCardState extends State<_ModalityCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final hasFile = widget.file != null;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            color: hasFile ? widget.color.withOpacity(0.06) : AppColors.bgCard,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: hasFile
                  ? widget.color.withOpacity(0.5)
                  : (_hovered ? widget.color.withOpacity(0.3) : AppColors.border),
              width: hasFile ? 1.5 : 1,
            ),
            boxShadow: hasFile
                ? [BoxShadow(color: widget.color.withOpacity(0.1), blurRadius: 16)]
                : [],
          ),
          child: Stack(
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Top row: badge + status
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: widget.color.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            widget.modality,
                            style: TextStyle(
                              color: widget.color,
                              fontSize: 11,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ),
                        const Spacer(),
                        if (hasFile)
                          Icon(Icons.check_circle_rounded, color: widget.color, size: 18)
                        else
                          Icon(Icons.add_circle_outline_rounded, color: AppColors.textMuted, size: 18),
                      ],
                    ),
                    const SizedBox(height: 12),
                    // Image preview or placeholder
                    Expanded(
                      child: hasFile
                          ? ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.file(
                                widget.file!,
                                fit: BoxFit.cover,
                                width: double.infinity,
                              ),
                            )
                          : Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Container(
                                    width: 48,
                                    height: 48,
                                    decoration: BoxDecoration(
                                      color: widget.color.withOpacity(0.08),
                                      shape: BoxShape.circle,
                                    ),
                                    child: Icon(
                                      Icons.add_photo_alternate_rounded,
                                      color: widget.color.withOpacity(0.6),
                                      size: 24,
                                    ),
                                  ),
                                  const SizedBox(height: 10),
                                  Text(
                                    'Tap to upload',
                                    style: TextStyle(
                                      color: AppColors.textMuted,
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      widget.description,
                      style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 10,
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),
              // Remove button
              if (hasFile)
                Positioned(
                  top: 8,
                  right: 8,
                  child: GestureDetector(
                    onTap: widget.onRemove,
                    child: Container(
                      width: 24,
                      height: 24,
                      decoration: BoxDecoration(
                        color: AppColors.bg.withOpacity(0.8),
                        shape: BoxShape.circle,
                        border: Border.all(color: AppColors.border),
                      ),
                      child: const Icon(Icons.close_rounded, size: 14, color: AppColors.textSecondary),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
