# Usability & Documentation Updates - October 2025

## Overview

This document summarizes the comprehensive usability and documentation improvements made to the Network Inference Toolkit to make it more accessible, robust, and user-friendly.

## Updates Completed

### ✅ 1. CLI Enhancements

**Added to All CLI Scripts:**
- Clear, formatted docstrings with usage examples
- Improved `--help` output with real-world examples
- Support for multiple output formats (CSV, JSON, Parquet)
- Progress indicators and richer logging
- Config file support (JSON/YAML)
- Better error messages with actionable suggestions

**Affected Files:**
- `src/contagion/cli.py`
- `src/contagion/cli_complex.py`
- `src/contagion/cli_inference.py`
- `src/semantic/community_cli.py`
- `src/semantic/time_slice_cli.py`
- `src/semantic/phrase_cli.py`
- `src/semantic/actor_cli.py`
- `src/semantic/transformers_cli.py`
- `src/semantic/kg_cli.py`
- `src/semantic/visualize_cli.py`

### ✅ 2. Config File Support

**New Features:**
- Created `config_loader.py` modules for contagion and semantic pipelines
- Added `--config` argument to all CLIs
- CLI arguments override config file values
- Supports both JSON and YAML formats

**Example Config Files:**
- `examples/contagion_config.json` - Simple contagion simulation config
- `examples/complex_contagion_config.json` - Complex contagion config

**Usage:**
```bash
# Use config file
python -m src.contagion.cli --config config.json

# Override specific parameters
python -m src.contagion.cli --config config.json --beta 0.2
```

### ✅ 3. Documentation Enhancements

**README.md:**
- Added "Best Practices" section
  - Data preparation guidelines
  - Performance optimization tips
  - Network quality recommendations
  - Workflow recommendations
  - Common pitfalls to avoid
- Added comprehensive "Troubleshooting" section
  - Installation issues
  - Performance problems
  - Output format issues
  - Contagion simulation troubleshooting
  - Data format errors
  - How to get help

**CONTAGION.md:**
- Added config file support documentation
- Expanded "Best Practices" section
  - Model selection guidance
  - Parameter selection tips
  - Simulation best practices
  - Parameter inference strategies
  - Network integration guide
  - Validation and analysis recommendations
- Comprehensive "Troubleshooting" section
  - Simulation issues
  - Performance problems
  - Output issues
  - Network compatibility
  - Model-specific problems

### ✅ 4. End-to-End Workflow Notebook

**Created: `examples/end_to_end_workflow.ipynb`**

A complete, step-by-step tutorial demonstrating:
1. Data loading and preparation
2. Semantic network building
3. Network analysis (centrality, communities)
4. Contagion simulation (SI/SIS/SIR)
5. Parameter inference
6. Visualization
7. Model comparison

**Features:**
- Real code examples (runnable)
- Clear explanations at each step
- Visual outputs (plots, network diagrams)
- Interpretation guidance
- Next steps and resources

### ✅ 5. Error Handling & Validation

**Enhanced `src/contagion/cli.py`:**
- File validation (existence, format, emptiness)
- Parameter validation with helpful ranges
- Model-specific requirement checks
- Column name validation with suggestions
- Node ID range validation
- Informative error messages with tips

**Error Message Examples:**
```
Error: Column 'source' not found in edges CSV.
Available columns: ['src', 'dst', 'weight']
Tip: Use --source-col and --target-col to specify column names
```

```
Invalid beta value: 1.5. Must be between 0 and 1.
Tip: Try values between 0.01 (slow spread) and 0.5 (fast spread)
```

### ✅ 6. Output Format Flexibility

**All CLIs now support:**
- `--output-format csv` (default) - Easy to inspect
- `--output-format json` - Structured data
- `--output-format parquet` - Compact, fast for large datasets

**Example:**
```bash
python -m src.contagion.cli edges.csv --output-path results --output-format parquet
```

## Files Added

- `src/contagion/config_loader.py` - Config file loading utilities
- `src/semantic/config_loader.py` - Config file loading utilities
- `examples/contagion_config.json` - Example simple contagion config
- `examples/complex_contagion_config.json` - Example complex contagion config
- `examples/end_to_end_workflow.ipynb` - Complete workflow tutorial
- `CHANGELOG_USABILITY.md` - This file

## Files Modified

### Major Updates:
- `README.md` - Added Best Practices & Troubleshooting (150+ lines)
- `CONTAGION.md` - Expanded with Best Practices & Troubleshooting (200+ lines)

### CLI Scripts (all updated with):
- Usage examples in docstrings
- Config file support
- Output format options
- Better error handling

## Breaking Changes

**None.** All changes are backward compatible:
- Existing command-line usage continues to work
- New arguments are optional
- Config files are opt-in
- Default behavior unchanged

## Migration Guide

### For Existing Users:

**No changes required.** Your existing scripts and commands will continue to work exactly as before.

**To adopt new features:**

1. **Use config files for complex workflows:**
   ```bash
   # Before
   python -m src.contagion.cli edges.csv --model sir --beta 0.2 --gamma 0.1 --timesteps 100 --seed 42
   
   # After (create config.json)
   python -m src.contagion.cli --config config.json
   ```

2. **Save results in different formats:**
   ```bash
   # Before (only CSV)
   python -m src.contagion.cli edges.csv > results.txt
   
   # After (structured output)
   python -m src.contagion.cli edges.csv --output-path results --output-format json
   ```

3. **Get better error messages:**
   - Errors now include suggestions and tips automatically
   - No action needed, just benefit from clearer messages

## Testing

**All existing tests pass:**
- ✅ 35/35 unit tests
- ✅ CI/CD pipeline passing
- ✅ No regressions introduced

**New functionality tested:**
- Config file loading (JSON and YAML)
- Output format options (CSV, JSON, Parquet)
- Error validation and messages

## Documentation Status

| Document | Status | Notes |
|----------|--------|-------|
| README.md | ✅ Updated | Added Best Practices & Troubleshooting |
| CONTAGION.md | ✅ Updated | Expanded with detailed guidance |
| CLI --help | ✅ Updated | All CLIs have improved help text |
| Examples | ✅ Created | Config files and workflow notebook |
| API Docs | ⏳ In Progress | Docstring improvements ongoing |

## Next Steps

### Recommended Actions:

1. **Review the new workflow notebook** (`examples/end_to_end_workflow.ipynb`)
   - Run it on your own data
   - Use it as a template for analyses

2. **Try config files** for complex workflows
   - Reduces command-line clutter
   - Makes workflows reproducible
   - Easy to version control

3. **Explore output formats**
   - Use Parquet for large-scale results
   - Use JSON for API integration
   - Use CSV for quick inspection

4. **Read the troubleshooting sections**
   - Common issues and solutions documented
   - Performance optimization tips
   - Best practices for each use case

### Ongoing Work:

- [ ] Expand docstrings for all public API functions
- [ ] Add more example notebooks (time-series, multilingual, etc.)
- [ ] Create video tutorials for common workflows
- [ ] Build interactive documentation website

## Feedback

Found an issue or have a suggestion? Please:
1. Check the Troubleshooting sections first
2. Review example notebooks for reference
3. Open an issue on GitHub with details

## Acknowledgments

These updates incorporate feedback from:
- User issue reports
- Documentation gaps identified in testing
- Best practices from the NetworkX and spaCy communities
- Accessibility and usability guidelines

---

**Summary:** The toolkit is now significantly more user-friendly, with comprehensive documentation, better error handling, flexible output options, and real-world examples. All changes are backward compatible, and existing code continues to work without modification.
