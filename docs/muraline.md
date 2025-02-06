# MuraLine Project at The University of British Columbia

## Dates
- **Start**: September 2024
- **End**: April 2025

## Links
- Portfolio Link: [[Portfolio Link](https://www.evanrichardsonengineering.com/work/muraline)]

## Summary
MuraLine is a **cable-driven mural-blocking system** designed to assist artists by automating the scaling and outlining process of large-scale murals. By precisely translating digital designs onto physical walls, the system reduces the time spent on scaffolding, minimizes labor costs, and streamlines the mural preparation process. The project leverages **stepper-driven motion control, computer vision-based image processing, and optimized path planning algorithms** to ensure accurate and efficient outlines on walls of various sizes and textures.

## Technologies Used
- **C++ (PlatformIO, Repetier Firmware)**
- **Python (OpenCV Image Processing, Path Planning)**
- **Java (SVG Parsing & UI Development)**
- **Stepper Motor Control & G-Code Processing**
- **Cable-Driven Gantry System Integration**

## Details
MuraLine was conceived to address the inefficiencies and technical barriers in large-scale mural creation. Currently, muralists rely on projection techniques, grid layouts, or freehand scaling methods to transfer designs onto walls. These methods can be time-consuming, physically demanding, and costly. MuraLine automates this process using a **dual-stage motion system**: a **macro cable-driven positioning system** that moves the painting head, and a **micro gantry system** for fine adjustments and application control.

The **macro motion system** utilizes Kevlar cables, mounted pulleys, and high-torque stepper motors to move the paint applicator over the wall surface. The **micro motion system** refines positioning accuracy, ensuring precise outline placement by compensating for positional drift or wall surface variations. 

### **Evan's Contributions**
Evan is leading the **software and firmware development** for MuraLine, ensuring seamless integration between the digital design pipeline and the physical painting system.

- **Firmware Development (Repetier 3D Printer Firmware)**
  - Modified **Repetier firmware** to control the motion of the cable-driven positioning system.
  - Configured **G-Code processing** for path execution with real-time motor control.
  - Integrated **Z-axis actuation logic** to allow height adjustments for uneven surfaces.

- **Software Development (SVG Parsing & Path Planning)**
  - Developed a **Java-based application** to parse SVG artwork files and extract contour lines for outlining.
  - Implemented **advanced path-planning algorithms** to optimize paint stroke continuity and minimize motion redundancy.
  - Designed an **interactive UI** that allows artists to **scale, position, and preview** mural outlines before execution.

- **Computer Vision Integration (OpenCV)**
  - Integrated **OpenCV-based image stitching** to scan and reconstruct the target wall surface before painting.
  - Developed **edge detection algorithms** to identify key mural features and improve outline accuracy.
  - Created an adaptive **obstruction avoidance system**, allowing the system to detect and avoid painting over windows, vents, or architectural features.

- **System Calibration & Testing**
  - Developed **automated calibration routines** to ensure accurate stepper motor synchronization.
  - Implemented a **feedback loop** for real-time trajectory corrections.
  - Designed and executed **simulation tests** to validate paint application precision before full-scale deployment.

### **Project Impact**
MuraLine has the potential to **revolutionize large-scale mural creation** by providing a cost-effective, automated solution for blocking out designs. By reducing **setup time, scaffolding dependency, and labor costs**, the system enables artists to focus on the creative aspects of mural production while ensuring precision in large-scale artwork.

By democratizing access to large-scale mural creation, MuraLine aims to **empower artists, reduce physical labor requirements, and introduce efficiency into the world of public art**.
