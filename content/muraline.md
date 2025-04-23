---
type: project
title: "MuraLine Project"
date_start: "2024-09"
date_end: "2025-04"
skills: [Motion Control, Path Planning, Computer Vision, Firmware Development, UI Development]
technologies: [C++, Python, Java, OpenCV, Repetier Firmware, PlatformIO, Stepper Motors, G-Code]
keywords: [Mural Automation, Cable-Driven Robotics, Image Processing, Motion Planning]
---

# MuraLine Project

## Overview
Developing **MuraLine**, a **cable-driven mural-blocking system** that automates **scaling and outlining** large-scale murals. The system translates **digital designs onto physical walls**, reducing **scaffolding time, labor costs, and mural preparation effort**.

- Designed a **dual-stage motion system** for macro and micro positioning accuracy.  
- Implemented **computer vision algorithms** to enhance **design placement and surface adaptation**.  
- Developed **firmware and software for motion control, SVG parsing, and G-Code execution**.  

## Technical Details
Built a **high-precision motion control system** integrating **firmware, computer vision, and UI-based planning tools**.

- **Cable-Driven Motion System**:  
  - High-torque **stepper motors control Kevlar cables**, enabling **large-scale positioning**.  
  - Integrated **micro-gantry system** for fine adjustments and **outline accuracy**.  

- **Firmware & Motion Control**:  
  - Modified **Repetier firmware** to control **stepper-driven cable positioning**.  
  - Configured **G-Code processing** for real-time execution of mural paths.  
  - Implemented **Z-axis actuation logic** for uneven surface adjustments.  

- **SVG Parsing & Path Planning**:  
  - Developed a **Java-based parser** to extract **contour lines from vector artwork**.  
  - Designed **path-planning algorithms** to optimize paint stroke efficiency.  
  - Built an **interactive UI** for **scaling, positioning, and previewing outlines** before execution.  

- **Computer Vision for Accuracy & Obstacle Detection**:  
  - Integrated **OpenCV-based image stitching** to scan and reconstruct the **wall surface**.  
  - Applied **edge detection and obstruction mapping** to avoid **windows, vents, and architectural features**.  
  - Developed a **real-time trajectory correction system** for increased precision.  

## Skills Demonstrated
Engineered a **multidisciplinary system combining robotics, vision processing, and UI design**.

- **Motion Control & Path Planning**: Built **stepper-driven motion algorithms for mural execution**.  
- **Firmware Development**: Modified **Repetier firmware for cable-driven motion control**.  
- **Computer Vision & Image Processing**: Implemented **OpenCV-based scanning and edge detection**.  
- **UI & Software Development**: Created an **SVG-based design parser with an interactive preview tool**.  

## Quantitative Outcomes
- **Increased efficiency**: Reduced **manual projection and grid setup time by 60%**.  
- **Improved accuracy**: Developed **trajectory correction algorithms**, ensuring **precise mural outlines**.  
- **Scalability**: Designed the system to work with **various wall sizes and surfaces**.  

## Additional Context
- **Project Scope**: Developing a **cost-effective, automated tool for muralists**.  
- **Future Enhancements**:  
  - **Integrating AI-based stroke prediction** for dynamic path refinement.  
  - **Expanding material compatibility** for different paint types and surfaces.  
  - **Deploying real-world trials with mural artists** to refine usability.  
