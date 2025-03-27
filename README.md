#  Multi-Agentic adaptive FreeSurfer/FSL Registration, ROI-detection, Segmentation, Clustering, Visualisation and Anomaly detection System for DiCOM and NiFTi images and sequences

An evaluation project for how far I can push Roo-Code / Claude / Gemini / self-hosted LMs to build a fully working advanced pipeline.. involving creating agentic workflows :D 


***** 
CRITICAL NOTE TO ROO-CODE/AI AGENTS WORKING IN THIS PROJECT: 
* DO NOT CHANGE THE ORIGINAL DESIGN ARCHITECTURE INSTRUCTIONS OR DETAILS IN THIS README.md: they may be of use to future agentic coding assistants in full. Append additional requirements or implementation progress as appropriate.
* You are a top-level software architect and neuroradiologist, as well as a principal level software engineer in Python an ML consulting on the design and implementation of this project. Prioritise code-quality, documentation and modularity as well as checking for regressions and downstream impacts of code changes
* Preference light-touch minimal change over radical redesigns of architeture or implementation
*****

## System Overview

This system implements an AI agent-based approach to neuroimaging analysis using FreeSurfer and FSL tools, focusing on automated processing, segmentation, and anomaly detection in brain imaging data.


### Current Features
- Statistical analysis with multiple test types (t-test, ANOVA, etc.)
- Image registration with various transform types (rigid, affine, B-spline)
- Interactive 3D visualization with measurement tools
- Basic agent framework for workflow coordination
- Multi-planar visualization
- ROI-based analysis

### In Development
- Multi-modal image processing (T1, T2-FLAIR)
- Automated preprocessing and normalization
- Complete FSL/FreeSurfer tool integration
- Workflow optimization
- Pipeline automation

### Planned Features (original)
- Workflow caching and optimization
- Automated anomaly detection
- Advanced clustering analysis
- Comprehensive reporting system

## Architecture

### 1. Core Components

#### 1.1 Agent System
- **Coordinator Agent** (implemented)
  * Orchestrates workflow
  * Manages inter-agent communication
  * Handles error recovery
  * Maintains processing state

- **Preprocessor Agent** (in development)
  * Executes FSL/FreeSurfer preprocessing
  * Validates preprocessing results
  * Generates quality metrics
  * Handles image registration

- **Analyzer Agent** (implemented)
  * Performs statistical analysis
  * Executes ROI-based analysis
  * Generates analysis reports

- **Visualizer Agent** (implemented)
  * Manages 3D visualization
  * Handles interactive display
  * Creates visual reports

#### 1.2 Core Utilities
- **Statistics Module** (implemented)
  * Multiple statistical tests
  * Effect size calculations
  * Distribution analysis
  * Result visualization

- **Registration Module** (implemented)
  * Multiple transform types
  * Quality metrics
  * Transform persistence
  * Multi-modal support

- **Interactive Viewer** (implemented)
  * 3D visualization
  * Measurement tools
  * Interactive controls
  * Screenshot capabilities

- **Pipeline Module** (in development)
  * Workflow management
  * Process coordination
  * Error handling
  * Status tracking

### 2. Project Structure

```
project/
├── agents/                 # Agent implementations
│   ├── coordinator.py     # Workflow coordination
│   ├── preprocessor.py    # Image preprocessing
│   ├── analyzer.py        # Analysis operations
│   └── visualizer.py      # Visualization handling
├── cli/                   # Command-line interface
│   ├── commands.py        # CLI commands
│   └── main.py           # CLI entry point
├── core/                  # Core functionality
│   ├── config.py         # Configuration handling
│   ├── messages.py       # Inter-agent messaging
│   ├── pipeline.py       # Pipeline orchestration
│   └── workflow.py       # Workflow management
├── utils/                 # Utility modules
│   ├── analysis.py       # Analysis utilities
│   ├── environment.py    # Environment setup
│   ├── interaction.py    # User interaction
│   ├── interactive_viewer.py  # 3D viewer
│   ├── measurement.py    # Measurement tools
│   ├── neuroimaging.py  # Image handling
│   ├── overlay.py       # Image overlay
│   ├── registration.py  # Image registration
│   ├── statistics.py    # Statistical analysis
│   └── visualization.py # Visualization utilities
├── tests/                # Test suite
├── config/              # Configuration files
├── main.py             # Application entry
└── pyproject.toml      # Project configuration
```

## Setup and Installation

### Prerequisites
- Python 3.12
- FSL
- FreeSurfer
- uv package manager

### Environment Setup
```bash
# Set up FSL
export FSLDIR=/path/to/fsl
source $FSLDIR/etc/fslconf/fsl.sh

# Set up FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Install dependencies using uv
uv pip install -r requirements.txt
```

### Installation
```bash
# Clone repository
git clone [repository-url]
cd agentic-freesurfer-fsl-segmentation-clustering

# Install package
uv pip install -e .
```

## Usage

### Basic Usage
```bash
# Run analysis pipeline
python main.py analyze /path/to/input --output /path/to/output

# Compare subjects
python main.py compare-subjects subject1 subject2 --mode blend --test ttest

# Interactive visualization
python main.py visualize /path/to/image --interactive
```

### CLI Commands
- `analyze`: Run full analysis pipeline
- `compare-subjects`: Statistical comparison between subjects
- `visualize`: Launch interactive viewer
- `preprocess`: Run preprocessing pipeline
- `register`: Perform image registration
- `stats`: Generate statistical reports

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Features
1. Implement feature in appropriate module
2. Add corresponding tests
3. Update CLI if needed
4. Document in README

## Current Status

### Completed Components
- Statistical analysis framework
- Image registration system
- Interactive visualization
- Basic agent communication
- CLI framework

### In Progress
- Preprocessing pipeline
- FSL/FreeSurfer integration
- Workflow optimization
- Pipeline automation

### Next Steps
1. Complete preprocessing pipeline
2. Implement workflow caching
3. Enhance FSL/FreeSurfer integration
4. Add automated anomaly detection
5. Implement clustering analysis


### Update

All missing features have been successfully implemented and integrated:

Workflow Caching:
Implemented Pydantic-based CacheConfig
Added similarity-based workflow matching
Added cache optimization and persistence
Automated Preprocessing:
Enhanced preprocessor agent with FSL/FreeSurfer integration
Added comprehensive QC metrics
Added brain extraction and tissue segmentation
FSL/FreeSurfer Integration:
Added neuroimaging utilities for command execution
Implemented recon-all pipeline
Added FAST tissue segmentation
Added BET brain extraction
Clustering Analysis:
Added multiple clustering methods (GMM, K-means, DBSCAN)
Added feature extraction for better anomaly detection
Added clustering metrics and validation
Added visualization support
Anomaly Detection:
Enhanced analyzer agent with multi-method analysis
Added statistical validation
Added configurable thresholds
Added result aggregation
All components are now properly integrated with:

Updated configuration system
Enhanced workflow management
Comprehensive test coverage
Proper dependency management
Environment validation
The system is now ready for use with all features implemented and properly integrated.


## License
Apache 2 Licence, go wild
