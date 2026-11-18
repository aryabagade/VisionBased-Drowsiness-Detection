import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPORTS_DIR = r"C:\Users\hp\.gemini\antigravity\scratch\VisionBased-Drowsiness-Detection\reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.6, "AI-Based Real-Time Driver Drowsiness & Motion Monitoring Architecture",
            fontsize=14, fontweight='bold', ha='center', color='#1A365D')
    ax.text(6, 6.2, "Department of Artificial Intelligence & Data Science | VIT Pune",
            fontsize=10, fontstyle='italic', ha='center', color='#4A5568')

    # Step styles
    box_blue = dict(boxstyle='round,pad=0.5', facecolor='#EBF8FF', edgecolor='#3182CE', linewidth=1.8)
    box_green = dict(boxstyle='round,pad=0.5', facecolor='#F0FFF4', edgecolor='#38A169', linewidth=1.8)
    box_red = dict(boxstyle='round,pad=0.5', facecolor='#FFF5F5', edgecolor='#E53E3E', linewidth=2.0)
    box_orange = dict(boxstyle='round,pad=0.5', facecolor='#FFFAF0', edgecolor='#DD6B20', linewidth=1.8)

    # 1. Video Input
    ax.text(1.2, 4.5, "1. Video Capture\nLaptop Webcam\n(1280x720 @ 30 FPS)", ha='center', va='center', bbox=box_blue, fontsize=8.5, fontweight='bold')
    ax.annotate('', xy=(2.4, 4.5), xytext=(2.0, 4.5), arrowprops=dict(facecolor='#3182CE', edgecolor='#3182CE', arrowstyle='->', lw=2))

    # 2. MediaPipe Landmark Mesh
    ax.text(3.6, 4.5, "2. Face & Mesh\nMediaPipe 3D\n(468 Landmarks)", ha='center', va='center', bbox=box_blue, fontsize=8.5, fontweight='bold')

    # Fan out arrows to 3 streams
    ax.annotate('', xy=(5.2, 5.5), xytext=(4.6, 4.7), arrowprops=dict(facecolor='#3182CE', edgecolor='#3182CE', arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(5.2, 4.5), xytext=(4.8, 4.5), arrowprops=dict(facecolor='#3182CE', edgecolor='#3182CE', arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(5.2, 3.5), xytext=(4.6, 4.3), arrowprops=dict(facecolor='#3182CE', edgecolor='#3182CE', arrowstyle='->', lw=1.5))

    # Stream A: Deep Learning Eye Classifier
    ax.text(6.4, 5.5, "3A. PyTorch CNN\nDrowsyEyeNet (.pth)\nOpen vs Closed (99.8%)", ha='center', va='center', bbox=box_green, fontsize=8.5, fontweight='bold')

    # Stream B: Geometric Aspect Ratios (EAR & MAR)
    ax.text(6.4, 4.5, "3B. Geometry Ratios\nEAR (Eye Aspect Ratio)\nMAR (Yawn Ratio)", ha='center', va='center', bbox=box_green, fontsize=8.5, fontweight='bold')

    # Stream C: 3D Head Pose
    ax.text(6.4, 3.5, "3C. 3D Head Pose\ncv2.solvePnP Euler\nYaw, Pitch, Roll", ha='center', va='center', bbox=box_green, fontsize=8.5, fontweight='bold')

    # Combine to Temporal Engine
    ax.annotate('', xy=(8.0, 4.5), xytext=(7.6, 5.3), arrowprops=dict(facecolor='#38A169', edgecolor='#38A169', arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(8.0, 4.5), xytext=(7.6, 4.5), arrowprops=dict(facecolor='#38A169', edgecolor='#38A169', arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(8.0, 4.5), xytext=(7.6, 3.7), arrowprops=dict(facecolor='#38A169', edgecolor='#38A169', arrowstyle='->', lw=1.5))

    # 4. Temporal Duration Engine
    ax.text(9.0, 4.5, "4. Temporal Engine\nBlink Filter (<0.35s)\nClosure Timer (Te >= 1.5s)\nYawn Timer (>= 1.8s)", ha='center', va='center', bbox=box_orange, fontsize=8.5, fontweight='bold')
    ax.annotate('', xy=(10.2, 4.5), xytext=(9.9, 4.5), arrowprops=dict(facecolor='#DD6B20', edgecolor='#DD6B20', arrowstyle='->', lw=2))

    # 5. Decision Engine
    ax.text(11.0, 4.5, "5. Decision Engine\nNORMAL (Green)\nWARNING (Gold)\nDROWSY (Red)\nDISTRACTED (Orange)", ha='center', va='center', bbox=box_red, fontsize=8.2, fontweight='bold')

    # Lower output row: Alerts and Logging
    ax.annotate('', xy=(9.0, 2.0), xytext=(11.0, 3.6), arrowprops=dict(facecolor='#E53E3E', edgecolor='#E53E3E', arrowstyle='->', lw=2))
    ax.text(7.5, 1.8, "Active Alert System\nHardware Buzzer (winsound.Beep)\n+ Flashing Red HUD Banner", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF5F5', edgecolor='#E53E3E', lw=1.5), fontsize=8.5, fontweight='bold')

    ax.text(3.5, 1.8, "Automated Telemetry Logging\nlogs/driver_events.csv\nTimestamp, State, EAR, MAR, Head Pose", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#EDF2F7', edgecolor='#718096', lw=1.5), fontsize=8.5, fontweight='bold')
    ax.annotate('', xy=(5.2, 1.8), xytext=(6.2, 1.8), arrowprops=dict(arrowstyle='<-', lw=1.5, color='#718096', ls='--'))

    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "architecture_diagram.png"), dpi=300)
    plt.close()
    print("Architecture diagram created!")

generate_system_architecture_diagram()
