# ReMold Project at The University of British Columbia

## Dates
- **Start**: September 2022
- **End**: April 2023

## Links
- Portfolio Link: [[Portfolio Link](https://www.evanrichardsonengineering.com/work/remold)]

## Summary
ReMold is an adaptable mold-making system designed to revolutionize composite prototyping by reducing cost, waste, and time associated with traditional mold-making methods. The system utilizes a reconfigurable array of threaded pins controlled by a gantry-driven actuation system. This enables rapid mold adjustments without requiring new materials for each iteration, significantly improving sustainability in composite manufacturing. The project was supported by a $2,000 grant from the UBC Sustainable Projects Fund.

## Technologies Used
- **C++ (PlatformIO, Marlin Firmware)**
- **Java (STL to G-Code Conversion)**
- **Stepper Motor Control & G-Code Processing**
- **Mechanical & Electrical Systems Integration**

## Details
ReMold was designed to address the inefficiencies of traditional mold-making for composite parts, where one-off molds are machined at high expense and often discarded after a single use. Through extensive stakeholder interviews, including discussions with automotive manufacturers and film set designers, the team identified the need for a cost-effective, reusable molding solution.

The system consists of an array of threaded rods that adjust their heights to create a customizable mold surface. A silicone membrane supported by a wire mesh forms the mold’s top surface, ensuring smooth and adaptable shapes. The pins are actuated via a motorized gantry system, controlled through G-Code generated from CAD models.

### **Evan's Contributions**
Evan was primarily responsible for **firmware development, software integration, and electrical system implementation**:
- **Firmware Development (Marlin 3D Printer Firmware)**
  - Modified and configured **Marlin firmware** to control the precise motion of the pin actuation system.
  - Implemented **custom boot sequences** and adapted **servo motor controls** to enable accurate Z-axis height adjustments.
  - Integrated limit switches and motion control parameters to ensure precision during mold formation.

- **Software Development (STL to G-Code Conversion)**
  - Created a **custom Java application** that converts STL files into depth maps, then translates them into **G-Code instructions** for controlling the threaded rods.
  - Implemented **file parsing algorithms** to process both ASCII and binary STL files.
  - Designed **filtering and transformation utilities** to optimize model resolution and ensure accurate mold geometry.

- **Electrical & Mechanical Integration**
  - Wired and assembled the **stepper motor drivers, limit switches, and control board** for the gantry system.
  - Contributed to **gantry installation, calibration, and movement testing** to ensure smooth motion and reliability.
  - Assisted with manufacturing, assembly, and composite layup testing.

### **Software Implementation & Testing**
- The software workflow **translates CAD models into G-Code commands**, allowing users to upload an STL file, select a desired mold face, and automatically generate height adjustments for each pin.
- A **test suite** was developed to validate software reliability, achieving **100% branch coverage** and **>90% line coverage** in automated verification.
- The **user experience was streamlined using Pronterface**, a common G-Code sender, to allow easy communication with the machine.

### **Project Impact**
ReMold successfully demonstrated the feasibility of **a reconfigurable, low-waste molding solution** for composite prototyping. By eliminating material waste and reducing production costs, this system offers a **sustainable alternative** to traditional mold manufacturing..

ReMold has the potential to significantly **reduce prototyping costs and improve sustainability** in industries reliant on composite manufacturing, making it a valuable innovation in mold-making technology.
