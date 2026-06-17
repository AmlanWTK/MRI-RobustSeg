// ignore_for_file: prefer_const_constructors, prefer_const_literals_to_create_immutables

import 'package:brain_tumor_detection/main.dart';
import 'package:brain_tumor_detection/screens/upload_screen.dart';
import 'package:brain_tumor_detection/services/ai_services.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';


// ─── Home Screen ─────────────────────────────────────────────────────────────
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _floatController;
  late Animation<double> _pulseAnim;
  late Animation<double> _floatAnim;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);

    _floatController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat(reverse: true);

    _pulseAnim = Tween<double>(begin: 0.85, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    _floatAnim = Tween<double>(begin: -8.0, end: 8.0).animate(
      CurvedAnimation(parent: _floatController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _floatController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(),
            Expanded(
              child: SingleChildScrollView(
                physics: const BouncingScrollPhysics(),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 16),
                      _buildHeroSection(),
                      const SizedBox(height: 32),
                      _buildStatsRow(),
                      const SizedBox(height: 32),
                      _buildSectionLabel('CAPABILITIES'),
                      const SizedBox(height: 16),
                      _buildCapabilityGrid(),
                      const SizedBox(height: 32),
                      _buildSectionLabel('MODEL SPECIFICATIONS'),
                      const SizedBox(height: 16),
                      _buildModelSpecs(),
                      const SizedBox(height: 32),
                      _buildStartButton(),
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

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        children: [
          // Logo dot
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: AppColors.accentPrimary,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(color: AppColors.accentPrimary.withOpacity(0.7), blurRadius: 8, spreadRadius: 2),
              ],
            ),
          ),
          const SizedBox(width: 10),
          Text(
            'TUMORLENS',
            style: TextStyle(
              color: AppColors.accentPrimary,
              fontSize: 13,
              fontWeight: FontWeight.w800,
              letterSpacing: 3,
            ),
          ),
          const Spacer(),
          GestureDetector(
            onTap: () => _showSystemInfoDialog(context),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.accentPrimary.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.accentPrimary.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  Icon(Icons.hub_rounded, size: 14, color: AppColors.accentPrimary),
                  const SizedBox(width: 6),
                  Text(
                    'SYSTEM',
                    style: TextStyle(
                      color: AppColors.accentPrimary,
                      fontSize: 10,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeroSection() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF0A1628),
            const Color(0xFF0D1F35),
          ],
        ),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: AppColors.accentPrimary.withOpacity(0.05),
            blurRadius: 40,
            spreadRadius: 0,
          ),
        ],
      ),
      child: Stack(
        children: [
          // Background grid pattern
          Positioned.fill(
            child: CustomPaint(painter: _GridPainter()),
          ),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: AppColors.accentPrimary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: AppColors.accentPrimary.withOpacity(0.3)),
                      ),
                      child: Text(
                        'POWERED BY U-Net++ · BraTS2020',
                        style: TextStyle(
                          color: AppColors.accentPrimary,
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.5,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Brain Tumor\nDetection',
                      style: TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 30,
                        fontWeight: FontWeight.w800,
                        height: 1.15,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Clinical-grade AI segmentation\nacross 3 tumor sub-regions',
                      style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 14,
                        height: 1.5,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 20),
              // Animated brain icon
              AnimatedBuilder(
                animation: _floatAnim,
                builder: (_, __) => Transform.translate(
                  offset: Offset(0, _floatAnim.value),
                  child: AnimatedBuilder(
                    animation: _pulseAnim,
                    builder: (_, __) => Transform.scale(
                      scale: _pulseAnim.value,
                      child: Container(
                        width: 90,
                        height: 90,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              AppColors.accentPrimary.withOpacity(0.2),
                              AppColors.accentPrimary.withOpacity(0.05),
                            ],
                          ),
                          border: Border.all(
                            color: AppColors.accentPrimary.withOpacity(0.4),
                            width: 1.5,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: AppColors.accentPrimary.withOpacity(0.3),
                              blurRadius: 24,
                              spreadRadius: 4,
                            ),
                          ],
                        ),
                        child: const Icon(
                          Icons.psychology_rounded,
                          size: 44,
                          color: AppColors.accentPrimary,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatsRow() {
    final stats = [
      _StatData('87.8%', 'Dice Score (WT)', AppColors.accentPrimary),
      _StatData('7.7M', 'Parameters', AppColors.info),
      _StatData('10.4mm', 'HD95 (TC)', AppColors.whole),
    ];

    return Row(
      children: stats.asMap().entries.map((e) {
        final isLast = e.key == stats.length - 1;
        return Expanded(
          child: Container(
            margin: EdgeInsets.only(right: isLast ? 0 : 12),
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
            decoration: BoxDecoration(
              color: AppColors.bgCard,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              children: [
                Text(
                  e.value.value,
                  style: TextStyle(
                    color: e.value.color,
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  e.value.label,
                  style: TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 10,
                    letterSpacing: 0.5,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        );
      }).toList(),
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
          style: TextStyle(
            color: AppColors.textMuted,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 2.5,
          ),
        ),
      ],
    );
  }

  Widget _buildCapabilityGrid() {
    final caps = [
      _CapData(Icons.scatter_plot_rounded, 'Enhancing\nTumor', 'Sub-region ET', AppColors.enhancing),
      _CapData(Icons.hub_rounded, 'Tumor\nCore', 'Sub-region TC', AppColors.core),
      _CapData(Icons.blur_circular_rounded, 'Whole\nTumor', 'Sub-region WT', AppColors.whole),
      _CapData(Icons.insights_rounded, 'Dice & IoU\nMetrics', 'Performance', AppColors.accentPrimary),
    ];

    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.4,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      children: caps.map((c) => _CapabilityCard(data: c)).toList(),
    );
  }

  Widget _buildModelSpecs() {
    final specs = [
      ['Architecture', 'Attention U-Net++'],
      ['Training Dataset', 'BraTS2020'],
      ['Input Modalities', 'FLAIR · T1 · T1CE · T2'],
      ['Input Resolution', '240 × 240 px'],
      ['DSC — Whole Tumor', '0.8783'],
      ['DSC — Tumor Core', '0.7898'],
      ['DSC — Enhancing', '0.7356'],
      ['HD95 — Tumor Core', '10.38 mm (best)'],
    ];

    return Container(
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: specs.asMap().entries.map((e) {
          final isLast = e.key == specs.length - 1;
          return Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                child: Row(
                  children: [
                    Text(
                      e.value[0],
                      style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 13,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      e.value[1],
                      style: TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              if (!isLast)
                Divider(height: 0, color: AppColors.border),
            ],
          );
        }).toList(),
      ),
    );
  }

  Widget _buildStartButton() {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        PageRouteBuilder(
          pageBuilder: (_, anim, __) => UploadScreen(),
          transitionsBuilder: (_, anim, __, child) => FadeTransition(
            opacity: anim,
            child: SlideTransition(
              position: Tween<Offset>(begin: const Offset(0, 0.05), end: Offset.zero)
                  .animate(CurvedAnimation(parent: anim, curve: Curves.easeOut)),
              child: child,
            ),
          ),
        ),
      ),
      child: Container(
        width: double.infinity,
        height: 60,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: const LinearGradient(
            colors: [AppColors.accentGlow, AppColors.accentSecondary],
          ),
          boxShadow: [
            BoxShadow(
              color: AppColors.accentPrimary.withOpacity(0.35),
              blurRadius: 24,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.biotech_rounded, color: AppColors.bg, size: 22),
            const SizedBox(width: 12),
            Text(
              'Start New Analysis',
              style: TextStyle(
                color: AppColors.bg,
                fontSize: 16,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
            ),
            const SizedBox(width: 12),
            const Icon(Icons.arrow_forward_rounded, color: AppColors.bg, size: 20),
          ],
        ),
      ),
    );
  }
  void _showSystemInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          backgroundColor: Colors.transparent,
          insetPadding: const EdgeInsets.all(20),
          child: Container(
            width: double.infinity,
            constraints: const BoxConstraints(maxWidth: 500, maxHeight: 700),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AppColors.bgSurface,
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: AppColors.accentPrimary.withOpacity(0.1),
                  blurRadius: 30,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.hub_rounded, color: AppColors.accentPrimary, size: 28),
                          const SizedBox(width: 12),
                          const Text(
                            'System Architecture',
                            style: TextStyle(
                              color: AppColors.textPrimary,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      IconButton(
                        icon: const Icon(Icons.close_rounded, color: AppColors.textSecondary),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  Text(
                    'Attention U-Net++ with HDA',
                    style: TextStyle(
                      color: AppColors.accentSecondary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'This system utilizes an Attention U-Net++ architecture featuring nested skip connections and soft attention gates for precise multimodal brain tumor segmentation. To ensure clinical robustness, we incorporate a physics-based Hybrid Degradation Augmentation (HDA) pipeline simulating realistic MRI artifacts (Rician noise, ghosting, Gibbs ringing, and bias fields) during training.',
                    style: TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 14,
                      height: 1.5,
                    ),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    'Key Innovations',
                    style: TextStyle(
                      color: AppColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildBulletPoint('Backbone', '5-level encoder-decoder with dense skip connections.'),
                  _buildBulletPoint('Attention Gates', 'Soft spatial attention for selective feature recalibration.'),
                  _buildBulletPoint('HDA Pipeline', 'Simulates real-world MRI acquisition degradations.'),
                  _buildBulletPoint('Loss Function', 'Combined Dice Loss + Focal Loss for class imbalance.'),
                  const SizedBox(height: 24),
                  const Text(
                    'Performance Metrics (BraTS 2020)',
                    style: TextStyle(
                      color: AppColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    decoration: BoxDecoration(
                      color: AppColors.bgCard,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Column(
                      children: [
                        _buildMetricDialogRow('DSC - Whole Tumor (WT)', '0.8783', AppColors.accentPrimary),
                        const Divider(height: 1, color: AppColors.border),
                        _buildMetricDialogRow('DSC - Tumor Core (TC)', '0.7898', AppColors.info),
                        const Divider(height: 1, color: AppColors.border),
                        _buildMetricDialogRow('DSC - Enhancing Tumor (ET)', '0.7356', AppColors.enhancing),
                        const Divider(height: 1, color: AppColors.border),
                        _buildMetricDialogRow('HD95 - Tumor Core (TC)', '10.38 mm', AppColors.success),
                        const Divider(height: 1, color: AppColors.border),
                        _buildMetricDialogRow('HD95 - Whole Tumor (WT)', '25.27 mm', AppColors.whole),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () => Navigator.of(context).pop(),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.accentPrimary.withOpacity(0.1),
                        foregroundColor: AppColors.accentPrimary,
                        elevation: 0,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      child: const Text('Got it'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildBulletPoint(String title, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(top: 6, right: 10),
            width: 6,
            height: 6,
            decoration: const BoxDecoration(
              color: AppColors.accentPrimary,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: const TextStyle(color: AppColors.textSecondary, fontSize: 13, height: 1.5),
                children: [
                  TextSpan(text: '$title: ', style: const TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.bold)),
                  TextSpan(text: description),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricDialogRow(String label, String value, Color valueColor) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppColors.textSecondary, fontSize: 13)),
          Text(value, style: TextStyle(color: valueColor, fontSize: 14, fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

// ─── Helper Data Classes ──────────────────────────────────────────────────────
class _StatData {
  final String value, label;
  final Color color;
  const _StatData(this.value, this.label, this.color);
}

class _CapData {
  final IconData icon;
  final String title, subtitle;
  final Color color;
  const _CapData(this.icon, this.title, this.subtitle, this.color);
}

class _CapabilityCard extends StatefulWidget {
  final _CapData data;
  const _CapabilityCard({required this.data});

  @override
  State<_CapabilityCard> createState() => _CapabilityCardState();
}

class _CapabilityCardState extends State<_CapabilityCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: _hovered ? AppColors.bgCardHover : AppColors.bgCard,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: _hovered
                ? widget.data.color.withOpacity(0.5)
                : AppColors.border,
          ),
          boxShadow: _hovered
              ? [BoxShadow(color: widget.data.color.withOpacity(0.12), blurRadius: 16)]
              : [],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: widget.data.color.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(widget.data.icon, color: widget.data.color, size: 20),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.data.title,
                  style: TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    height: 1.3,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  widget.data.subtitle,
                  style: TextStyle(
                    color: widget.data.color,
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Grid Background Painter ──────────────────────────────────────────────────
class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.accentPrimary.withOpacity(0.04)
      ..strokeWidth = 0.5;

    const spacing = 24.0;
    for (double x = 0; x < size.width; x += spacing) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += spacing) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(_GridPainter old) => false;
}
