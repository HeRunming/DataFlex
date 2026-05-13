# DataFlex LESS Selector Documentation Index

This document provides a comprehensive index of all available documentation about the LESS selector implementation in DataFlex.

---

## 📚 Documentation Files

### 1. **LESS_SELECTOR_ANALYSIS.md** (Comprehensive Technical Deep-Dive)
**Purpose:** Detailed analysis of LESS implementation for developers who need to understand or modify the code.

**Contents:**
- Complete source code walkthrough with line references
- Each function documented with:
  - Purpose and role in pipeline
  - Input/output specifications
  - Code flow and logic
  - Key implementation details
  - How it integrates with distributed training
- Distributed training patterns explained in detail
- Memory and performance characteristics

**Best For:**
- Understanding how LESS works internally
- Implementing custom selectors based on LESS
- Debugging issues in selector logic
- Modifying or extending LESS functionality

**Read This If You Need:**
- Complete technical understanding of the code
- To answer questions like "what exactly happens in `_obtain_gradients()`?"
- To implement custom scoring or projection methods

---

### 2. **LESS_SELECTOR_QUICK_REFERENCE.md** (Quick Lookup Guide)
**Purpose:** Quick reference for developers who already understand LESS and need fast lookups.

**Contents:**
- Parameter descriptions (one-liners)
- Key values and their meanings
- Common configurations and presets
- Function signatures and returns
- Important code locations (file:line format)
- Typical values for different scenarios
- Memory and compute requirements
- Troubleshooting checklist

**Best For:**
- Quick parameter lookup
- Finding specific code locations
- Memory/performance ballpark estimates
- Tuning configurations for different hardware

**Read This If You Need:**
- Parameter definitions without lengthy explanations
- Code location reference
- Memory requirements for planning
- Common configuration templates

---

### 3. **LESS_SELECTOR_VISUAL_GUIDE.md** (Flowcharts and Diagrams)
**Purpose:** Visual representation of LESS selector execution and data flow.

**Contents:**
- ASCII flowcharts showing:
  - Complete selection pipeline flow
  - Per-GPU gradient collection process
  - Multi-GPU synchronization pattern
  - Training loop timeline and selection triggers
- Data structure visualizations
- Execution sequence diagrams
- Timeline showing what happens at each step
- Directory structure during execution

**Best For:**
- Understanding the big picture
- Presentations and explanations to others
- Understanding multi-GPU coordination
- Quick mental model of the system

**Read This If You Need:**
- High-level overview of how LESS works
- Visual representation of execution flow
- To understand distributed training patterns
- To present LESS to others

---

### 4. **LESS_SETUP_AND_CONFIGURATION.md** (Setup and Usage Guide)
**Purpose:** Complete guide to configuring and running LESS selector experiments.

**Contents:**
- Configuration file documentation:
  - components.yaml parameters explained
  - training config (less.yaml) walk-through
- Selection pipeline phases explained
- Distributed training mechanics
- Gradient processing details (computation, preconditioning, projection)
- Scoring and selection algorithm
- Integration with training loop
- Data configuration and dataset setup
- Running experiments (basic and advanced)
- Monitoring and debugging
- Memory considerations
- Comparison with other selectors
- Common issues and solutions
- Advanced customization techniques
- Quick reference templates

**Best For:**
- Setting up LESS for experiments
- Understanding configuration options
- Debugging training issues
- Running and monitoring experiments
- Understanding data flow

**Read This If You Need:**
- To run LESS experiments
- To understand what each config parameter does
- To troubleshoot training problems
- Memory planning for your hardware
- Dataset configuration guidance

---

## 🎯 Quick Start by Use Case

### I want to understand how LESS works
1. Start with **LESS_SELECTOR_VISUAL_GUIDE.md** (15 min read)
2. Read **LESS_SETUP_AND_CONFIGURATION.md** sections 1-5 (20 min read)
3. Deep dive with **LESS_SELECTOR_ANALYSIS.md** as needed

### I want to run a LESS experiment
1. Read **LESS_SETUP_AND_CONFIGURATION.md** sections 1-3 (15 min read)
2. Customize config based on section 8 (5 min)
3. Run experiment and monitor per section 9 (ongoing)

### I want to modify LESS implementation
1. Read **LESS_SELECTOR_ANALYSIS.md** completely (45 min read)
2. Use **LESS_SELECTOR_QUICK_REFERENCE.md** for code locations (5 min lookups)
3. Modify and test (time varies)

### I want to debug a LESS problem
1. Check **LESS_SETUP_AND_CONFIGURATION.md** section 11 for common issues
2. Use **LESS_SELECTOR_QUICK_REFERENCE.md** to find relevant code sections
3. Consult **LESS_SELECTOR_ANALYSIS.md** for implementation details
4. Monitor with guidance from section 9

### I need specific parameter information
- Use **LESS_SELECTOR_QUICK_REFERENCE.md** section 2 (Parameters)
- Cross-reference with **LESS_SETUP_AND_CONFIGURATION.md** for detailed explanations

### I need to estimate memory/compute requirements
- Use **LESS_SETUP_AND_CONFIGURATION.md** section 9.3
- Use **LESS_SELECTOR_QUICK_REFERENCE.md** section 8

---

## 📖 Information by Topic

### Configuration and Setup
| Topic | Primary Doc | Secondary Doc | Section |
|-------|-------------|---------------|---------|
| LESS parameters | QUICK_REF | SETUP | 1.1 / 2 |
| Training config | SETUP | VISUAL | 1.2 / - |
| Components.yaml | SETUP | QUICK_REF | 1.1 / 2 |
| Dataset setup | SETUP | - | 7 |
| Running experiments | SETUP | QUICK_REF | 8 / 6 |

### Technical Details
| Topic | Primary Doc | Secondary Doc | Section |
|-------|-------------|---------------|---------|
| Selection pipeline | VISUAL | ANALYSIS | 2 / 1 |
| Gradient computation | ANALYSIS | SETUP | 1 / 4.1 |
| TRAK projection | ANALYSIS | SETUP | 2 / 4.3 |
| Multi-GPU sync | VISUAL | ANALYSIS | 3 / 3-5 |
| Scoring algorithm | ANALYSIS | SETUP | 4 / 5 |
| Caching mechanism | ANALYSIS | SETUP | 5 / 5.2 |

### Troubleshooting and Optimization
| Topic | Primary Doc | Secondary Doc | Section |
|-------|-------------|---------------|---------|
| Common issues | SETUP | QUICK_REF | 11 / 7 |
| Memory planning | SETUP | QUICK_REF | 9.3 / 8 |
| Performance tuning | QUICK_REF | SETUP | 8 / 12 |
| Debugging | QUICK_REF | ANALYSIS | 7 / 1-8 |
| Advanced customization | SETUP | ANALYSIS | 12 / 1-8 |

---

## 🔍 Finding Answers to Common Questions

### Q: What does each configuration parameter do?
**A:** Use **LESS_SELECTOR_QUICK_REFERENCE.md** section 2, then cross-reference with **LESS_SETUP_AND_CONFIGURATION.md** section 1

### Q: How are gradients computed and projected?
**A:** Read **LESS_SELECTOR_VISUAL_GUIDE.md** flowchart, then **LESS_SETUP_AND_CONFIGURATION.md** section 4

### Q: What's the exact sequence of operations during selection?
**A:** See **LESS_SELECTOR_VISUAL_GUIDE.md** for flowcharts, **LESS_SELECTOR_ANALYSIS.md** for code details

### Q: How does multi-GPU training work?
**A:** Read **LESS_SELECTOR_VISUAL_GUIDE.md** section "Multi-GPU Synchronization", then **LESS_SETUP_AND_CONFIGURATION.md** section 3

### Q: Where in the code is [specific thing]?
**A:** Use **LESS_SELECTOR_QUICK_REFERENCE.md** section 4 "Code Locations" or search in **LESS_SELECTOR_ANALYSIS.md**

### Q: How much memory will my experiment need?
**A:** Read **LESS_SETUP_AND_CONFIGURATION.md** section 9.3 "Memory Considerations"

### Q: What's wrong with my LESS experiment?
**A:** Check **LESS_SETUP_AND_CONFIGURATION.md** section 11 "Common Issues and Solutions"

### Q: How do I modify LESS for [specific use case]?
**A:** See **LESS_SETUP_AND_CONFIGURATION.md** section 12 "Advanced Customization"

### Q: What optimizer states are available?
**A:** See **LESS_SETUP_AND_CONFIGURATION.md** section 6.1 and **LESS_SELECTOR_ANALYSIS.md** section on `_prepare_optimizer_state()`

### Q: How often does selection happen?
**A:** Read **LESS_SETUP_AND_CONFIGURATION.md** sections 1.2 and 6.2 "Training Timeline"

---

## 📊 Documentation Coverage Matrix

| Topic | ANALYSIS | QUICK_REF | VISUAL | SETUP |
|-------|----------|-----------|--------|-------|
| **Architecture** | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| **Configuration** | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| **Implementation Details** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| **Usage/Setup** | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| **Debugging** | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| **Performance** | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| **Visual Diagrams** | ☆☆☆☆☆ | ☆☆☆☆☆ | ★★★★★ | ★★☆☆☆ |
| **Code References** | ★★★★★ | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ |

Legend: ★ = coverage level (1-5 stars)

---

## 🚀 Reading Paths by Experience Level

### Beginner (New to LESS and DataFlex)
**Total Time: ~90 minutes**

1. **LESS_SELECTOR_VISUAL_GUIDE.md** (15 min)
   - Understand the big picture
   - See how data flows through the system

2. **LESS_SETUP_AND_CONFIGURATION.md** sections 1-3 (20 min)
   - Understand configuration structure
   - Learn what each parameter does
   - Understand the three phases of selection

3. **LESS_SETUP_AND_CONFIGURATION.md** sections 8 (15 min)
   - See how to run an experiment
   - Understand the commands

4. **LESS_SETUP_AND_CONFIGURATION.md** sections 9-11 (30 min)
   - Learn how to monitor experiments
   - See common issues and their solutions

5. **LESS_SELECTOR_QUICK_REFERENCE.md** (10 min)
   - Bookmark for future reference

### Intermediate (Familiar with DataFlex, new to LESS)
**Total Time: ~60 minutes**

1. **LESS_SELECTOR_VISUAL_GUIDE.md** (10 min)
   - Get visual understanding of execution flow

2. **LESS_SETUP_AND_CONFIGURATION.md** sections 1-6 (25 min)
   - Deep understanding of configuration
   - How selection integrates with training loop

3. **LESS_SELECTOR_QUICK_REFERENCE.md** (15 min)
   - Parameter reference
   - Code location reference

4. **LESS_SETUP_AND_CONFIGURATION.md** sections 9-11 (10 min)
   - Monitoring and debugging

### Advanced (Need to modify LESS implementation)
**Total Time: ~120 minutes**

1. **LESS_SELECTOR_QUICK_REFERENCE.md** (10 min)
   - Get code location map

2. **LESS_SELECTOR_ANALYSIS.md** (60 min)
   - Complete implementation understanding
   - Line-by-line walkthrough

3. **LESS_SELECTOR_VISUAL_GUIDE.md** (15 min)
   - Understand execution flow visually

4. **LESS_SETUP_AND_CONFIGURATION.md** sections 1, 12 (20 min)
   - Configuration structure
   - Customization techniques

5. **LESS_SELECTOR_QUICK_REFERENCE.md** (15 min)
   - Bookmark for future code lookups

---

## 📝 Notes on Documentation Maintenance

These documents were created from:
- Source code analysis: `less_selector.py`, `select_trainer.py`, `selector_io.py`, `base_selector.py`
- Configuration files: `components.yaml`, `less.yaml`
- Dataset registry: `dataset_info.json`

**When to update documentation:**
- Gradient computation changes → Update ANALYSIS and SETUP sections 4
- Configuration schema changes → Update QUICK_REF and SETUP sections 1
- Multi-GPU logic changes → Update VISUAL and ANALYSIS sections 3-5
- Scoring algorithm changes → Update ANALYSIS section 4 and SETUP section 5

---

## 🎓 Learning Outcomes by Document

### After reading LESS_SELECTOR_VISUAL_GUIDE.md, you will understand:
- What are the three main phases of LESS selection
- How data flows from training through selection to next epoch
- What happens on each GPU during selection
- How rank 0 (main process) coordinates multi-GPU selection
- The timeline of when selection happens during training

### After reading LESS_SELECTOR_QUICK_REFERENCE.md, you will know:
- All configuration parameters and what they control
- Where each major function is located in the code
- Typical values for different scenarios
- Memory requirements for different dataset sizes
- Quick troubleshooting checklist

### After reading LESS_SELECTOR_ANALYSIS.md, you will understand:
- Exactly what each function does (line by line)
- How distributed training patterns are implemented
- Memory allocation and usage patterns
- How to add custom functionality
- Integration points with the rest of DataFlex

### After reading LESS_SETUP_AND_CONFIGURATION.md, you will be able to:
- Configure LESS for your specific use case
- Run LESS experiments on any number of GPUs
- Debug common issues
- Monitor and optimize experiments
- Add custom datasets
- Estimate hardware requirements
- Customize gradient computation and scoring

---

## 🔗 Cross-References Between Documents

These four documents are designed to work together. Common reference points:

- **Configuration parameters**: Defined in SETUP 1.1, referenced in QUICK_REF 2, visualized in VISUAL, implemented in ANALYSIS
- **Gradient flow**: Shown in VISUAL, explained in SETUP 4 and ANALYSIS 1-2, configured in QUICK_REF 2
- **Multi-GPU coordination**: Visualized in VISUAL 3, implemented in ANALYSIS 3-5, used in SETUP 3 and 6
- **Selection pipeline**: Shown in VISUAL 2.1, explained in SETUP 2, implemented in ANALYSIS 4-6
- **Debugging**: Issues listed in SETUP 11, locations in QUICK_REF 7, implementation in ANALYSIS 1-8

---

## 📞 Getting Help

### For questions about...

| Question Type | Best Resource | Backup Resources |
|---------------|----------------|------------------|
| Configuration | SETUP 1 | QUICK_REF 2 |
| Code location | QUICK_REF 4 | ANALYSIS (search) |
| How it works | VISUAL | ANALYSIS |
| How to run | SETUP 8 | QUICK_REF 6 |
| What's wrong | SETUP 11 | QUICK_REF 7 |
| How to modify | SETUP 12 | ANALYSIS |
| Memory planning | SETUP 9.3 | QUICK_REF 8 |

---

## Summary

These four documents provide comprehensive coverage of the LESS selector from multiple angles:

- **VISUAL**: Architecture and flow (best for understanding the big picture)
- **QUICK_REF**: Parameters and code locations (best for quick lookup)
- **ANALYSIS**: Implementation details (best for deep understanding)
- **SETUP**: Configuration and usage (best for running experiments)

Start with the document matching your current goal, then cross-reference with others as needed!
