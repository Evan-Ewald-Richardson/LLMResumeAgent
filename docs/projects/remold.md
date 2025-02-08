---
type: project
title: "ReMold Project"
date_start: "2022-09"
date_end: "2023-04"
skills: [Firmware Development, Motion Control, CAD to G-Code Conversion, Mechanical Systems Integration, Electrical Engineering]
technologies: [C++, Java, Marlin Firmware, Stepper Motors, STL Parsing, G-Code Processing]
keywords: [Adaptive Molding, Composite Manufacturing, Sustainable Engineering, Motion Control]
---

# ReMold Project

## Overview
Developed **ReMold**, a **reconfigurable mold-making system** designed to reduce **cost, waste, and setup time** in composite prototyping. The system uses **threaded pin arrays controlled by a gantry-driven actuation system**, enabling rapid mold adjustments without requiring new materials for each iteration.

- Eliminated **one-off mold production waste** by using a **reusable pin-actuated mold surface**.  
- Built **a motion control system and software pipeline** for automated mold shaping.  
- Secured **$2,000 in funding** from the **UBC Sustainable Projects Fund**.  

## Technical Details
Designed a **motorized, software-driven mold system** integrating **firmware, CAD conversion, and mechanical actuation**.

- **Pin-Based Actuation System**:  
  - Controlled **threaded rods via stepper motors**, adjusting pin heights to form mold surfaces.  
  - Supported **smooth silicone membrane and wire mesh overlay** for adaptable shaping.  

- **Firmware & Motion Control (Marlin Firmware)**:  
  - Customized **Marlin firmware** to control stepper-driven pin actuation.  
  - Implemented **Z-axis height adjustment algorithms** for precise mold shaping.  
  - Integrated **limit switches and motion constraints** for calibration and accuracy.  

- **Software Development (STL to G-Code Processing)**:  
  - Developed a **Java-based STL parser** to convert **CAD models into G-Code commands**.  
  - Implemented **depth map extraction and transformation utilities** to optimize mold accuracy.  
  - Created **an automated workflow** for generating **height adjustments from CAD input**.  

## Skills Demonstrated
Applied **firmware development, software engineering, and mechanical integration** to automate mold shaping.

- **Firmware Engineering**: Modified **Marlin firmware for real-time motion control**.  
- **Software Development**: Built **STL-to-G-Code translation software** for mold automation.  
- **Mechanical & Electrical Integration**: Designed **gantry system electronics, motor drivers, and control logic**.  

## Quantitative Outcomes
- **Zero-waste mold reuse**: Eliminated the need for **disposable molds**, reducing **material waste**.  
- **High precision control**: Implemented **automated calibration routines**, improving **mold repeatability**.  
- **100% branch test coverage**: Developed **automated validation for STL processing software**.  

## Additional Context
- **Project Motivation**: Addressed **high costs and waste in composite mold prototyping**.  
- **Future Enhancements**:  
  - **Integrate AI-driven mold optimization** for complex geometries.  
  - **Expand mold compatibility** for different composite materials.  
  - **Develop a user-friendly GUI** for real-time mold adjustments.  
