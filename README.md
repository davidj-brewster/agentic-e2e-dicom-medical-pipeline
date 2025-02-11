# Agentic FreeSurfer/FSL Segmentation and Clustering System

[Previous content remains unchanged until Test Coverage]

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