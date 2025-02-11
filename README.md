# Agentic FreeSurfer/FSL Segmentation and Clustering System

## System Overview

This system implements an AI agent-based approach to neuroimaging analysis using FreeSurfer and FSL tools, focusing on automated processing, segmentation, and anomaly detection in brain imaging data.

### Key Features
- Multi-modal image processing (T1, T2-FLAIR)
- Automated preprocessing and normalization
- Region-specific segmentation
- Intensity-based clustering analysis
- Workflow optimization through prompt caching
- 3D visualization of anomalies

### FSL Integration
- Use `fslmaths` for image manipulation
- Implement `FIRST` for subcortical segmentation
- Utilize `FLIRT/FNIRT` for registration
- Apply `fslstats` for statistical analysis

### FreeSurfer Integration
- Leverage `recon-all` pipeline components
- Use `mri_convert` for format conversion
- Implement `mri_segment` for tissue classification
- Utilize `freeview` for visualization

## Implementation Guide

This guide outlines the optimal sequence for implementing the neuroimaging analysis system using Roo Code assistant, with clear validation points and minimal rework.

### Phase 1: Environment and Core Setup

1. **Environment Validation Script**
   ```python
   # Prompt: "Create a script that validates FreeSurfer and FSL installations, checking for required binaries and environment variables"
   ```
   - Validation: Script should return detailed status of environment setup
   - Files: `utils/environment.py`
   - Test: Run script on system with/without proper setup

2. **Test Data Preparation**
   ```python
   # Prompt: "Create a script to generate or download minimal test data for development"
   ```
   - Validation: Script should provide sample T1/T2-FLAIR data
   - Files: `tests/data/generate_test_data.py`
   - Test: Verify generated data format and structure

### Phase 2: Core Data Structures

1. **Message Protocol**
   ```python
   # Prompt: "Implement the base message protocol for inter-agent communication with pydantic models"
   ```
   - Validation: Unit tests for message serialization/deserialization
   - Files: `core/messages.py`
   - Test: Verify message validation and conversion

2. **Workflow Definitions**
   ```python
   # Prompt: "Create workflow step definitions and state management classes"
   ```
   - Validation: Unit tests for workflow state transitions
   - Files: `core/workflow.py`
   - Test: Verify workflow step sequencing

### Phase 3: Agent Framework

1. **Base Agent**
   ```python
   # Prompt: "Implement the base agent class with core messaging and state management"
   ```
   - Validation: Unit tests for basic agent functionality
   - Files: `agents/base.py`
   - Test: Verify agent lifecycle and messaging

2. **Mock Testing Framework**
   ```python
   # Prompt: "Create a mock framework for testing agent interactions"
   ```
   - Validation: Example tests using mock framework
   - Files: `tests/mocks.py`
   - Test: Verify agent interaction simulation

### Phase 4: Individual Agents

1. **Coordinator Agent**
   ```python
   # Prompt: "Implement the coordinator agent with workflow management"
   ```
   - Validation: Tests for workflow coordination
   - Files: `agents/coordinator.py`
   - Test: Verify workflow initialization and monitoring

2. **Preprocessor Agent**
   ```python
   # Prompt: "Create the preprocessor agent with FSL integration"
   ```
   - Validation: Tests with mock FSL commands
   - Files: `agents/preprocessor.py`
   - Test: Verify preprocessing workflow

3. **Analyzer Agent**
   ```python
   # Prompt: "Implement the analyzer agent with segmentation and clustering"
   ```
   - Validation: Tests with sample segmentation data
   - Files: `agents/analyzer.py`
   - Test: Verify analysis pipeline

4. **Visualizer Agent**
   ```python
   # Prompt: "Create the visualizer agent with multi-planar and 3D visualization"
   ```
   - Validation: Tests generating sample visualizations
   - Files: `agents/visualizer.py`
   - Test: Verify visualization outputs

### Phase 5: Pipeline Integration

1. **Pipeline Orchestrator**
   ```python
   # Prompt: "Implement the main pipeline orchestrator with agent coordination"
   ```
   - Validation: Integration tests with mock agents
   - Files: `core/pipeline.py`
   - Test: Verify end-to-end workflow

2. **Workflow Cache**
   ```python
   # Prompt: "Create the workflow caching system with Anthropic API integration"
   ```
   - Validation: Tests for cache storage/retrieval
   - Files: `core/cache.py`
   - Test: Verify workflow optimization

### Phase 6: CLI and Configuration

1. **Configuration System**
   ```python
   # Prompt: "Implement the configuration management system"
   ```
   - Validation: Tests for config loading/validation
   - Files: `core/config.py`
   - Test: Verify configuration handling

2. **CLI Interface**
   ```python
   # Prompt: "Create the command-line interface with rich progress display"
   ```
   - Validation: CLI usage examples
   - Files: `main.py`
   - Test: Verify command-line operation

### Phase 7: Documentation and Examples

1. **API Documentation**
   ```python
   # Prompt: "Generate comprehensive API documentation with examples"
   ```
   - Validation: Documentation build
   - Files: `docs/`
   - Test: Verify documentation completeness

2. **Usage Examples**
   ```python
   # Prompt: "Create example scripts for common use cases"
   ```
   - Validation: Example execution
   - Files: `examples/`
   - Test: Verify example functionality

## Validation Strategy

### Unit Testing
- Each component should have comprehensive unit tests
- Use pytest fixtures for test data and mocks
- Implement test coverage reporting

### Integration Testing
- Test agent interactions with mock framework
- Verify workflow transitions
- Test FSL/FreeSurfer integration

### System Testing
- End-to-end tests with sample data
- Performance benchmarking
- Error handling verification

## Development Tips

1. **Independent Development**
   - Each component can be developed and tested independently
   - Use mock objects for dependencies
   - Implement clear interfaces between components

2. **Incremental Testing**
   - Write tests before implementation
   - Verify each component in isolation
   - Integration tests for component combinations

3. **Documentation**
   - Document design decisions
   - Include validation criteria
   - Provide usage examples

4. **Error Handling**
   - Implement comprehensive error handling
   - Include recovery mechanisms
   - Log important events

## Implementation Sequence

1. Start with environment validation
2. Implement core data structures
3. Create base agent framework
4. Develop individual agents
5. Integrate pipeline components
6. Add CLI and configuration
7. Complete documentation

This sequence minimizes rework by:
- Establishing core components first
- Using mock objects for testing
- Implementing clear interfaces
- Validating each step independently

## Verification Points

Each phase should be verified by:
1. Unit tests passing
2. Integration tests successful
3. Documentation complete
4. Example usage working
5. Error handling verified

## Optimization Opportunities

1. **Parallel Development**
   - Multiple components can be developed simultaneously
   - Independent testing possible
   - Clear interfaces enable integration

2. **Incremental Deployment**
   - System can be tested with partial functionality
   - Add features progressively
   - Validate each addition

3. **Performance Tuning**
   - Profile each component
   - Optimize critical paths
   - Cache frequently used data

## Next Steps

1. Begin with environment validation script
2. Implement core message protocol
3. Create base agent framework
4. Develop agents incrementally
5. Integrate components
6. Add CLI and documentation

This approach ensures:
- Minimal rework required
- Independent component validation
- Clear progress tracking
- Maintainable codebase


### Agent Communication Protocol
```python
# Example agent message structure
{
    "message_id": str,
    "sender": str,
    "recipient": str,
    "message_type": str,
    "payload": {
        "command": str,
        "parameters": dict,
        "status": str,
        "data": Any
    },
    "timestamp": datetime,
    "priority": int
}
```

### Workflow Caching
```python
# Example cache structure
{
    "workflow_id": str,
    "subject_id": str,
    "pipeline_steps": [
        {
            "step_id": str,
            "tool": str,
            "parameters": dict,
            "success_metrics": dict,
            "prompt_template": str,
            "completion_tokens": int
        }
    ],
    "performance_metrics": dict,
    "timestamp": datetime
}

## Architecture

### 1. Core Components

#### 1.1 Data Input Handler
- Supports NIfTI (.nii) and DICOM formats
- Validates input data structure and completeness
- Implements modality detection and sorting
- Performs initial quality checks

#### 1.2 Preprocessing Pipeline
- Resolution normalization using `fslmaths`
- Orientation standardization
- Registration of T2-FLAIR to T1 space
- Quality control metrics generation

#### 1.3 Segmentation Engine
- Implements `run_first_all` for anatomical segmentation
- Generates binary masks per region
- Maintains segmentation quality metrics
- Handles multi-modal integration

#### 1.4 Clustering Analysis
- Region-specific intensity analysis
- Outlier detection in T2-FLAIR space
- Statistical analysis of cluster characteristics
- Anomaly scoring and classification

#### 1.5 Visualization Module
- Multi-planar (sagittal, coronal, axial) views
- 3D rendering via `freeview`
- Anomaly highlighting and reporting
- Interactive visualization controls

#### 1.6 Workflow Cache

- Stores successful processing pipelines
- Implements Anthropic API for prompt caching
- Maintains processing metadata
- Enables workflow optimization

### 2. Agent System Design

#### 2.1 Agent Types
1. **Coordinator Agent**
   - Orchestrates overall workflow
   - Manages inter-agent communication
   - Handles error recovery
   - Maintains processing state

2. **Preprocessing Agent**
   - Executes FSL/FreeSurfer preprocessing commands
   - Validates preprocessing results
   - Generates quality metrics
   - Handles image registration

3. **Analysis Agent**
   - Performs segmentation
   - Executes clustering algorithms
   - Identifies anomalies
   - Generates analysis reports

4. **Visualization Agent**
   - Manages `freeview` integration
   - Generates visualization outputs
   - Handles interactive display requests
   - Creates summary reports

#### 2.2 Agent Communication
- Asynchronous message passing
- Shared state management
- Progress monitoring
- Error handling protocols

### 3. Data Flow

```
Input Data
    │
    ▼
Validation & Preprocessing
    │
    ▼
Registration & Normalization
    │
    ▼
Segmentation
    │
    ▼
Clustering Analysis
    │
    ▼
Anomaly Detection
    │
    ▼
Visualization & Reporting
```

### 4. Technical Implementation

#### 4.1 Core Technologies
- Python 3.12
- FreeSurfer/FSL libraries
- Anthropic Claude API
- NumPy/SciPy for numerical processing

#### 4.2 Directory Structure
```
project/
├── agents/
│   ├── coordinator.py
│   ├── preprocessor.py
│   ├── analyzer.py
│   └── visualizer.py
├── core/
│   ├── data_handler.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── clustering.py
├── utils/
│   ├── cache.py
│   ├── visualization.py
│   └── metrics.py
├── config/
│   ├── pipeline_config.py
│   └── agent_config.py
└── tests/
```

#### 4.3 Configuration Management
- FSL/FreeSurfer environment variables
- Processing pipeline parameters
- Agent behavior settings
- Caching preferences

### 5. Workflow Optimization

#### 5.1 Prompt Caching Strategy
- Cache successful preprocessing sequences
- Store effective segmentation parameters
- Maintain modality-specific optimizations
- Enable rapid workflow adaptation

#### 5.2 Performance Considerations
- Parallel processing where applicable
- Resource utilization monitoring
- Memory management for large datasets
- Disk space optimization

### 6. Error Handling

#### 6.1 Recovery Mechanisms
- Automatic retry logic
- Graceful degradation
- State preservation
- Error logging and reporting

#### 6.2 Quality Control
- Input data validation
- Processing step verification
- Output quality metrics
- Result consistency checks

### Test Coverage Design

1. **Command Tests**
   ```python
   class TestCompareSubjectsCommand:
       """Tests for subject comparison command"""
       @pytest.mark.asyncio
       async def test_basic_comparison(
           self,
           test_subjects: List[Dict[str, Path]],
           mock_config: PipelineConfig
       ):
           """Test basic comparison functionality"""
           # Test blend mode
           args = ["compare-subjects"] + [s["id"] for s in test_subjects]
           exit_code = await main(args)
           assert exit_code == 0
           
           # Test different modes
           for mode in OverlayMode:
               args = [
                   "compare-subjects",
                   *[s["id"] for s in test_subjects],
                   "--mode", mode.name.lower()
               ]
               exit_code = await main(args)
               assert exit_code == 0
           
           # Test different tests
           for test in StatisticalTest:
               args = [
                   "compare-subjects",
                   *[s["id"] for s in test_subjects],
                   "--test", test.name.lower()
               ]
               exit_code = await main(args)
               assert exit_code == 0
   ```

2. **Statistical Tests**
   ```python
   @pytest.mark.asyncio
   async def test_statistical_analysis(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test statistical analysis"""
       # Create output directory
       output_dir = mock_config.output_dir / "statistical_test"
       
       # Test with different statistical tests
       for test in StatisticalTest:
           args = [
               "compare-subjects",
               *[s["id"] for s in test_subjects],
               "--test", test.name.lower(),
               "--output", str(output_dir)
           ]
           exit_code = await main(args)
           assert exit_code == 0
           
           # Verify outputs
           assert (output_dir / "statistics.json").exists()
           assert (output_dir / "distributions.png").exists()
           assert (output_dir / "correlation.png").exists()
           assert (output_dir / "report.html").exists()
           
           # Verify statistics
           with open(output_dir / "statistics.json") as f:
               stats = json.load(f)
           
           assert "roi_statistics" in stats
           assert "test_results" in stats
           assert test.name.lower() in stats["test_results"]
           
           result = stats["test_results"][test.name.lower()]
           assert "statistic" in result
           assert "p_value" in result
           assert "effect_size" in result
   ```

3. **ROI Tests**
   ```python
   @pytest.mark.asyncio
   async def test_roi_analysis(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test ROI analysis"""
       # Create output directory
       output_dir = mock_config.output_dir / "roi_test"
       
       # Test with ROI masks
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--output", str(output_dir)
       ]
       exit_code = await main(args)
       assert exit_code == 0
       
       # Verify statistics
       with open(output_dir / "statistics.json") as f:
           stats = json.load(f)
       
       assert "roi_statistics" in stats
       for subject in test_subjects:
           assert subject["id"] in stats["roi_statistics"]
       2    subject_stats = stats["roi_statistics"][subject["id"]]
           assert "mean" in subject_stats
           assert "std" in subject_stats
           assert "volume" in subject_stats
           assert "skewness" in subject_stats
           assert "kurtosis" in subject_stats
           assert "iqr" in subject_stats
   ```

4. **Interactive Tests**
   ```python
   @pytest.mark.asyncio
   async def test_interactive_mode(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test interactive mode"""
       # Create output directory
       output_dir = mock_config.output_dir / "interactive_test"
       
       # Test interactive mode
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--interactive",
           "--output", str(output_dir)
       ]
       
       # Mock tkinter
       with patch("tkinter.Tk") as mock_tk:
           # Mock window
           mock_window = MagicMock()
           mock_tk.return_value = mock_window
           
           # Mock notebook
           mock_notebook = MagicMock()
           mock_window.children["!notebook"] = mock_notebook
           
           # Execute command
           exit_code = await main(args)
           assert exit_code == 0
           
           # Verify window creation
           mock_tk.assert_called_once()
           
           # Verify notebook creation
           assert len(mock_notebook.tabs()) == 2
           assert "Visualization" in mock_notebook.tab(0)["text"]
           assert "Statistics" in mock_notebook.tab(1)["text"]
           
           # Verify controls
           assert len(mock_window.children) > 0
           assert any(
               isinstance(child, ttk.Button)
               for child in mock_window.children.values()
           )

           ### Test Coverage Design

1. **Command Tests**
   ```python
   class TestCompareSubjectsCommand:
       """Tests for subject comparison command"""
       @pytest.mark.asyncio
       async def test_basic_comparison(
           self,
           test_subjects: List[Dict[str, Path]],
           mock_config: PipelineConfig
       ):
           """Test basic comparison functionality"""
           # Test blend mode
           args = ["compare-subjects"] + [s["id"] for s in test_subjects]
           exit_code = await main(args)
           assert exit_code == 0
           
           # Test different modes
           for mode in OverlayMode:
               args = [
                   "compare-subjects",
                   *[s["id"] for s in test_subjects],
                   "--mode", mode.name.lower()
               ]
               exit_code = await main(args)
               assert exit_code == 0
           
           # Test different tests
           for test in StatisticalTest:
               args = [
                   "compare-subjects",
                   *[s["id"] for s in test_subjects],
                   "--test", test.name.lower()
               ]
               exit_code = await main(args)
               assert exit_code == 0
   ```

2. **Statistical Tests**
   ```python
   @pytest.mark.asyncio
   async def test_statistical_analysis(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test statistical analysis"""
       # Create output directory
       output_dir = mock_config.output_dir / "statistical_test"
       
       # Test with different statistical tests
       for test in StatisticalTest:
           args = [
               "compare-subjects",
               *[s["id"] for s in test_subjects],
               "--test", test.name.lower(),
               "--output", str(output_dir)
           ]
           exit_code = await main(args)
           assert exit_code == 0
           
           # Verify outputs
           assert (output_dir / "statistics.json").exists()
           assert (output_dir / "distributions.png").exists()
           assert (output_dir / "correlation.png").exists()
           assert (output_dir / "report.html").exists()
           
           # Verify statistics
           with open(output_dir / "statistics.json") as f:
               stats = json.load(f)
           
           assert "roi_statistics" in stats
           assert "test_results" in stats
           assert test.name.lower() in stats["test_results"]
           
           result = stats["test_results"][test.name.lower()]
           assert "statistic" in result
           assert "p_value" in result
           assert "effect_size" in result
   ```

3. **ROI Tests**
   ```python
   @pytest.mark.asyncio
   async def test_roi_analysis(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test ROI analysis"""
       # Create output directory
       output_dir = mock_config.output_dir / "roi_test"
       
       # Test with ROI masks
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--output", str(output_dir)
       ]
       exit_code = await main(args)
       assert exit_code == 0
       
       # Verify statistics
       with open(output_dir / "statistics.json") as f:
           stats = json.load(f)
       
       assert "roi_statistics" in stats
       for subject in test_subjects:
           assert subject["id"] in stats["roi_statistics"]
           subject_stats = stats["roi_statistics"][subject["id"]]
           assert "mean" in subject_stats
           assert "std" in subject_stats
           assert "volume" in subject_stats
           assert "skewness" in subject_stats
           assert "kurtosis" in subject_stats
           assert "iqr" in subject_stats
   ```

4. **Interactive Tests**
   ```python
   @pytest.mark.asyncio
   async def test_interactive_mode(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test interactive mode"""
       # Create output directory
       output_dir = mock_config.output_dir / "interactive_test"
       
       # Test interactive mode
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--interactive",
           "--output", str(output_dir)
       ]
       
       # Mock tkinter
       with patch("tkinter.Tk") as mock_tk:
           # Mock window
           mock_window = MagicMock()
           mock_tk.return_value = mock_window
           
           # Mock notebook
           mock_notebook = MagicMock()
           mock_window.children["!notebook"] = mock_notebook
           
           # Execute command
           exit_code = await main(args)
           assert exit_code == 0
           
           # Verify window creation
           mock_tk.assert_called_once()
           
           # Verify notebook creation
           assert len(mock_notebook.tabs()) == 2
           assert "Visualization" in mock_notebook.tab(0)["text"]
           assert "Statistics" in mock_notebook.tab(1)["text"]
           
           # Verify controls
           assert len(mock_window.children) > 0
           assert any(
               isinstance(child, ttk.Button)
               for child in mock_window.children.values()
           )
   ```

### Test Implementation Steps

1. **Basic Tests**
   - Command arguments
   - Mode selection
   - Test selection
   - Output handling
   - Result validation

2. **Statistical Tests**
   - Test execution
   - Result validation
   - Output validation
   - Format validation
   - Error handling

3. **ROI Tests**
   - Mask handling
   - Statistics calculation
   - Result validation
   - Output validation
   - Error handling

4. **Interactive Tests**
   - Window creation
   - Control handling
   - Event handling
   - Result saving
   - Error handling

### Test Development Approach

1. **Core Testing**
   - Command tests
   - Statistical tests
   - ROI tests
   - Interactive tests
   - Unit tests

2. **Integration Testing**
   - Command flow
   - Data handling
   - File management
   - Error handling
   - System tests

3. **UI Testing**
   - Window creation
   - Control handling
   - Event handling
   - Result display
   - Visual tests

4. **Validation**
   - Input validation
   - Output validation
   - Format validation
   - Performance testing
   - System testing

This approach ensures:
- Comprehensive coverage
- Robust validation
- Clear feedback
- Reliable results
- Thorough testing
```

### Test Implementation Steps

1. **Basic Tests**
   - Command arguments
   - Mode selection
   - Test selection
   - Output handling
   - Result validation

2. **Statistical Tests**
   - Test execution
   - Result validation
   - Output validation
   - Format validation
   - Error handling

3. **ROI Tests**
   - Mask handling
   - Statistics calculation
   - Result validation
   - Output validation
   - Error handling

4. **Interactive Tests**
   - Window creation
   - Control handling
   - Event handling
   - Result saving
   - Error handling

### Test Development Approach

1. **Core Testing**
   - Command tests
   - Statistical tests
   - ROI tests
   - Interactive tests
   - Unit tests

2. **Integration Testing**
   - Command flow
   - Data handling
   - File management
   - Error handling
   - System tests

3. **UI Testing**
   - Window creation
   - Control handling
   - Event handling
   - Result display
   - Visual tests

4. **Validation**
   - Input validation
   - Output validation
   - Format validation
   - Performance testing
   - System testing

This approach ensures:
- Comprehensive coverage
- Robust validation
- Clear feedback
- Reliable results
- Thorough testing




The agentic FreeSurfer/FSL segmentation and clustering system has been implemented with the following structure:

```
project/
├── agents/
│   ├── coordinator.py    # Workflow management and agent coordination
│   ├── preprocessor.py   # FSL/FreeSurfer preprocessing operations
│   ├── analyzer.py       # Segmentation and clustering analysis
│   └── visualizer.py     # Multi-planar and 3D visualization
├── core/
│   └── pipeline.py       # Main pipeline orchestration
├── main.py              # CLI entry point
├── pyproject.toml       # Project configuration
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

To use the system:

1. Set up environment variables:
```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export FSLDIR=/path/to/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export ANTHROPIC_API_KEY=your_api_key
```

2. Install dependencies:
```bash
pip install -e .
```

3. Run the pipeline:
```bash
python main.py /path/to/dataset --output-dir /path/to/output
```

The system will:
- Process each subject in the dataset
- Generate segmentations and analyze clusters
- Create visualizations and reports
- Cache successful workflows for optimization

Output includes:
- Preprocessed images
- Segmentation masks
- Cluster analysis results
- Multi-planar visualizations
- 3D renderings
- HTML reports
- Processing summaries

The workflow is optimized through:
- Anthropic API-powered workflow caching
- Adaptive parameter tuning
- Inter-subject optimization

1. Current State:
- Statistical comparison implemented
- Registration working
- Visualization working
- Interactive controls working
- Result persistence working

2. Integration Requirements:
a) Command Interface:
   - Statistical test selection
   - ROI selection
   - Test parameters
   - Result format
   - Export options

b) Workflow:
   - Image loading
   - Registration
   - Statistics
   - Visualization
   - Export

c) User Experience:
   - Test selection
   - Parameter control
   - Result display
   - Interactive plots
   - Report generation

d) Result Management:
   - Statistics files
   - Plot files
   - Reports
   - Summaries
   - Archives

Let me update the implementation guide with the command integration design:

## Validation Strategy

### Unit Testing
- Each component should have comprehensive unit tests
- Use pytest fixtures for test data and mocks
- Implement test coverage reporting

### Integration Testing
- Test agent interactions with mock framework
- Verify workflow transitions
- Test FSL/FreeSurfer integration

### System Testing
- End-to-end tests with sample data
- Performance benchmarking
- Error handling verification

## Development Tips

1. **Independent Development**
   - Each component can be developed and tested independently
   - Use mock objects for dependencies
   - Implement clear interfaces between components

2. **Incremental Testing**
   - Write tests before implementation
   - Verify each component in isolation
   - Integration tests for component combinations

3. **Documentation**
   - Document design decisions
   - Include validation criteria
   - Provide usage examples

4. **Error Handling**
   - Implement comprehensive error handling
   - Include recovery mechanisms
   - Log important events

## Implementation Sequence

1. Start with environment validation
2. Implement core data structures
3. Create base agent framework
4. Develop individual agents
5. Integrate pipeline components
6. Add CLI and configuration
7. Complete documentation

This sequence minimizes rework by:
- Establishing core components first
- Using mock objects for testing
- Implementing clear interfaces
- Validating each step independently

## Verification Points

Each phase should be verified by:
1. Unit tests passing
2. Integration tests successful
3. Documentation complete
4. Example usage working
5. Error handling verified

## Optimization Opportunities

1. **Parallel Development**
   - Multiple components can be developed simultaneously
   - Independent testing possible
   - Clear interfaces enable integration

2. **Incremental Deployment**
   - System can be tested with partial functionality
   - Add features progressively
   - Validate each addition

3. **Performance Tuning**
   - Profile each component
   - Optimize critical paths
   - Cache frequently used data

## Next Steps

1. Begin with environment validation script
2. Implement core message protocol
3. Create base agent framework
4. Develop agents incrementally
5. Integrate components
6. Add CLI and documentation

This approach ensures:
- Minimal rework required
- Independent component validation
- Clear progress tracking
- Maintainable codebase

### Command Integration Design

1. **Command Enhancement**
   ```python
   class CompareSubjectsCommand(Command):
       """Compare multiple subjects command"""
       name = "compare-subjects"
       help = "Compare multiple subjects"
       arguments = [
           Argument(
               "subject_ids",
               help="Subject IDs (space-separated)",
               nargs="+",
               required=True
           ),
           Argument(
               "--mode",
               help="Comparison mode",
               choices=[m.name.lower() for m in OverlayMode],
               default="blend"
           ),
           Argument(
               "--registration",
               help="Registration type",
               choices=[t.name.lower() for t in TransformType],
               default="affine"
           ),
           Argument(
               "--test",
               help="Statistical test",
               choices=[t.name.lower() for t in StatisticalTest],
               default="ttest"
           ),
           Argument(
               "--alpha",
               help="Significance level",
               type=float,
               default=0.05
           ),
           Argument(
               "--interactive",
               help="Enable interactive mode",
               action="store_true"
           ),
           Argument(
               "--output",
               help="Output directory",
               type=Path,
               default=None
           )
       ]
   ```

2. **Workflow Integration**
   ```python
   async def execute(self, args: Namespace) -> None:
       """Execute comparison command"""
       try:
           # Initialize pipeline
           pipeline = Pipeline(self.config)
           
           # Create output directory
           output_dir = self.create_output_directory(args)
           
           # Load and register images
           images, masks = await self.load_and_register_images(
               pipeline,
               args.subject_ids,
               args.registration,
               output_dir
           )
           
           # Create comparison
           comparison = StatisticalComparison(
               images,
               masks,
               args.subject_ids
           )
           
           # Perform statistical test
           result = comparison.perform_statistical_test(
               StatisticalTest[args.test.upper()],
               args.alpha
           )
           
           if args.interactive:
               # Create interactive visualization
               await self.show_interactive_comparison(
                   comparison,
                   args.mode,
                   output_dir
               )
           else:
               # Generate static results
               comparison.save_results(output_dir)
           
           # Display results
           self.display_results(result, output_dir)
           
       except Exception as e:
           logger.error(f"Comparison failed: {e}")
   ```

3. **Result Display**
   ```python
   def display_results(
       self,
       result: StatisticalResult,
       output_dir: Path
   ) -> None:
       """Display statistical results"""
       logger.info("\nStatistical Results:")
       logger.info(f"Test: {result.test_type.name}")
       logger.info(f"Statistic: {result.statistic:.3f}")
       logger.info(f"P-value: {result.p_value:.3e}")
       logger.info(f"Effect size: {result.effect_size:.3f}")
       
       if result.confidence_interval:
           ci_low, ci_high = result.confidence_interval
           logger.info(
               f"95% CI: ({ci_low:.3f}, {ci_high:.3f})"
           )
       
       logger.info(f"\nResults saved to: {output_dir}")
       logger.info("Files:")
       logger.info("- statistics.json")
       logger.info("- distributions.png")
       logger.info("- correlation.png")
       logger.info("- report.html")
   ```

4. **Interactive Visualization**
   ```python
   async def show_interactive_comparison(
       self,
       comparison: StatisticalComparison,
       mode: str,
       output_dir: Path
   ) -> None:
       """Show interactive comparison"""
       import tkinter as tk
       from matplotlib.backends.backend_tkagg import (
           FigureCanvasTkAgg,
           NavigationToolbar2Tk
       )
       
       # Create window
       root = tk.Tk()
       root.title("Statistical Comparison")
       
       # Create notebook for tabs
       notebook = ttk.Notebook(root)
       notebook.pack(fill="both", expand=True)
       
       # Distribution tab
       dist_frame = ttk.Frame(notebook)
       notebook.add(dist_frame, text="Distributions")
       
       dist_fig = comparison.create_distribution_plot()
       canvas = FigureCanvasTkAgg(dist_fig, dist_frame)
       canvas.draw()
       canvas.get_tk_widget().pack(fill="both", expand=True)
       
       toolbar = NavigationToolbar2Tk(canvas, dist_frame)
       toolbar.update()
       
       # Correlation tab
       if len(comparison.images) == 2:
           corr_frame = ttk.Frame(notebook)
           notebook.add(corr_frame, text="Correlation")
           
           corr_fig = comparison.create_correlation_plot()
           canvas = FigureCanvasTkAgg(corr_fig, corr_frame)
           canvas.draw()
           canvas.get_tk_widget().pack(fill="both", expand=True)
           
           toolbar = NavigationToolbar2Tk(canvas, corr_frame)
           toolbar.update()
       
       # Statistics tab
       stats_frame = ttk.Frame(notebook)
       notebook.add(stats_frame, text="Statistics")
       
       stats = comparison.calculate_roi_statistics()
       for label, stat in stats.items():
           group_frame = ttk.LabelFrame(
               stats_frame,
               text=label
           )
           group_frame.pack(fill="x", padx=5, pady=5)
           
           for name, value in stat.items():
               ttk.Label(
                   group_frame,
                   text=f"{name}: {value:.3f}"
               ).pack(anchor="w")
       
       # Save button
       def save_results():
           comparison.save_results(output_dir)
           messagebox.showinfo(
               "Success",
               f"Results saved to: {output_dir}"
           )
       
       ttk.Button(
           root,
           text="Save Results",
           command=save_results
       ).pack(pady=10)
       
       # Start event loop
       root.mainloop()
   ```

### Implementation Steps

1. **Command Updates**
   - Add arguments
   - Add workflow
   - Add visualization
   - Add interaction
   - Write tests

2. **Workflow Integration**
   - Add registration
   - Add statistics
   - Add visualization
   - Add export
   - Write tests

3. **Result Management**
   - Add display
   - Add interaction
   - Add persistence
   - Add validation
   - Write tests

4. **System Testing**
   - Command testing
   - Workflow testing
   - UI testing
   - Export testing
   - Integration testing

### Development Approach

1. **Core Integration**
   - Command updates
   - Workflow handling
   - Result management
   - Error handling
   - Unit tests

2. **User Interface**
   - Interactive plots
   - Statistics display
   - Result saving
   - Error feedback
   - Visual tests

3. **Result Handling**
   - Statistics files
   - Plot files
   - Reports
   - Archives
   - Integration tests

4. **Validation**
   - Input validation
   - Result validation
   - Format validation
   - Performance testing
   - System testing

This approach ensures:
- Comprehensive analysis
- Clear visualization
- Intuitive interaction
- Reliable results
- Thorough testing<
```

### Implementation Steps

1. **Statistical Analysis**
   - Add ROI statistics
   - Add t-tests
   - Add effect size
   - Add correlation
   - Write tests

2. **Statistical Visualization**
   - Add distributions
   - Add correlations
   - Add plots
   - Add interactivity
   - Write tests

3. **Result Management**
   - Add statistics
   - Add plots
   - Add reports
   - Add export
   - Write tests

4. **Integration**
   - Update command
   - Add UI
   - Add validation
   - Add persistence
   - Write tests

### Development Approach

1. **Core Analysis**
   - Statistics
   - Tests
   - Effect sizes
   - Validation
   - Unit tests

2. **Visualization**
   - Plots
   - Interactivity
   - Export
   - Validation
   - Visual tests

3. **Reporting**
   - Statistics
   - Plots
   - Templates
   - Export
   - Integration tests

4. **System Testing**
   - Command testing
   - UI testing
   - Export testing
   - Performance testing
   - User testing

This approach ensures:
- Robust statistics
- Clear visualization
- Comprehensive reports
- Intuitive interaction
- Thorough testing

### Test Coverage Design

1. **Command Tests**
   ```python
   class TestCompareSubjectsCommand:
       """Tests for subject comparison command"""
       @pytest.mark.asyncio
       async def test_basic_comparison(
           self,
           test_subjects: List[Dict[str, Path]],
           mock_config: PipelineConfig
       ):
           """Test basic comparison functionality"""
           # Test blend mode
           args = ["compare-subjects"] + [s["id"] for s in test_subjects]
           exit_code = await main(args)
           assert exit_code == 0
           
           # Test different modes
           for mode in OverlayMode:
               args = [
                   "compare-subjects",
                   *[s["id"] for s in test_subjects],
                   "--mode", mode.name.lower()
               ]
               exit_code = await main(args)
               assert exit_code == 0
   ```

2. **Registration Tests**
   ```python
   @pytest.mark.asyncio
   async def test_registration(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test image registration"""
       # Test different transforms
       for transform in TransformType:
           args = [
               "compare-subjects",
               *[s["id"] for s in test_subjects],
               "--registration", transform.name.lower()
           ]
           exit_code = await main(args)
           assert exit_code == 0
           
           if mock_config.output_dir:
               # Verify transform files
               for subject in test_subjects[1:]:
                   transform_path = (
                       mock_config.output_dir /
                       f"{subject['id']}_transform.tfm"
                   )
                   assert transform_path.exists()
   ```

3. **Visualization Tests**
   ```python
   @pytest.mark.asyncio
   async def test_visualization(
       self,
       test_subjects: List[Dict[str, Path]],
       mock_config: PipelineConfig
   ):
       """Test visualization modes"""
       # Test interactive mode
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--interactive"
       ]
       exit_code = await main(args)
       assert exit_code == 0
       
       # Test static mode with output
       output_dir = mock_config.output_dir / "comparison"
       args = [
           "compare-subjects",
           *[s["id"] for s in test_subjects],
           "--output", str(output_dir)
       ]
       exit_code = await main(args)
       assert exit_code == 0
       
       # Verify outputs
       assert output_dir.exists()
       assert (output_dir / "comparison.png").exists()
   ```

4. **Error Handling Tests**
   ```python
   @pytest.mark.asyncio
   async def test_error_handling(
       self,
       mock_config: PipelineConfig
   ):
       """Test error handling"""
       # Test invalid subject
       args = ["compare-subjects", "nonexistent"]
       exit_code = await main(args)
       assert exit_code == 1
       
       # Test invalid mode
       args = [
           "compare-subjects",
           "test_subject",
           "--mode", "invalid"
       ]
       exit_code = await main(args)
       assert exit_code == 1
       
       # Test invalid registration
       args = [
           "compare-subjects",
           "test_subject",
           "--registration", "invalid"
       ]
       exit_code = await main(args)
       assert exit_code == 1
       
       # Test invalid output path
       args = [
           "compare-subjects",
           "test_subject",
           "--output", "/invalid/path"
       ]
       exit_code = await main(args)
       assert exit_code == 1
   ```

### Test Implementation Steps

1. **Basic Tests**
   - Command arguments
   - Mode selection
   - Metric selection
   - Output handling
   - Result validation

2. **Registration Tests**
   - Transform types
   - Success/failure
   - File saving
   - Error handling
   - Result validation

3. **Visualization Tests**
   - Overlay modes
   - Interactive mode
   - Static mode
   - Screenshot saving
   - Result validation

4. **Error Tests**
   - Invalid inputs
   - Missing data
   - File errors
   - System errors
   - Result validation

### Test Development Approach

1. **Core Testing**
   - Command tests
   - Registration tests
   - Visualization tests
   - Error tests
   - Unit tests

2. **Integration Testing**
   - Command flow
   - Data handling
   - File management
   - Error handling
   - System tests

3. **UI Testing**
   - Control creation
   - Event handling
   - View updates
   - Result display
   - Visual tests

4. **Validation**
   - Input validation
   - Output validation
   - Error validation
   - Performance testing
   - System testing

This approach ensures:
- Comprehensive coverage
- Robust validation
- Clear feedback
- Reliable results
- Thorough testing