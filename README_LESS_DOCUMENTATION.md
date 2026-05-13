# LESS Selector Documentation - Complete Guide

## 🎯 Introduction

This repository contains comprehensive documentation for the **LESS (Low-rank Embedding Selectivity Strategy)** selector in DataFlex. LESS is a sophisticated gradient-based data selection mechanism that dynamically selects training samples based on their estimated influence on model evaluation performance.

This package includes **6 complementary documentation files totaling ~100KB** that cover every aspect of LESS from high-level architecture to line-by-line implementation details.

---

## 📦 Documentation Package Contents

### Quick Reference
| File | Size | Purpose | Best For |
|------|------|---------|----------|
| **DOCUMENTATION_INDEX.md** | 15K | Navigation hub | Finding what you need |
| **LESS_SELECTOR_README.md** | 9.8K | Quick overview | First-time readers |

### Core Documentation (Choose Based on Your Goal)
| File | Size | Purpose | Best For |
|------|------|---------|----------|
| **LESS_SELECTOR_VISUAL_GUIDE.md** | 26K | Flowcharts and diagrams | Understanding architecture |
| **LESS_SELECTOR_QUICK_REFERENCE.md** | 13K | Parameter lookup | Quick information access |
| **LESS_SELECTOR_ANALYSIS.md** | 19K | Implementation deep-dive | Code understanding |
| **LESS_SETUP_AND_CONFIGURATION.md** | 17K | Setup and usage guide | Running experiments |

**Total: ~100KB of documentation**

---

## 🚀 Getting Started in 5 Minutes

### Step 1: Understand What You Want to Do

Choose ONE:
- "I want to **understand how LESS works**" → Start with LESS_SELECTOR_VISUAL_GUIDE.md
- "I want to **run a LESS experiment**" → Start with LESS_SETUP_AND_CONFIGURATION.md section 1-3
- "I want to **modify LESS code**" → Start with LESS_SELECTOR_ANALYSIS.md
- "I need to **quickly find something**" → Use DOCUMENTATION_INDEX.md as your map

### Step 2: Follow the Recommended Reading Path

**Beginner (~90 min):**
```
VISUAL_GUIDE (15 min) → SETUP 1-3 (20 min) → SETUP 8-11 (30 min) → QUICK_REF (10 min)
```

**Intermediate (~60 min):**
```
VISUAL_GUIDE (10 min) → SETUP 1-6 (25 min) → QUICK_REF (15 min) → SETUP 9-11 (10 min)
```

**Advanced (~120 min):**
```
QUICK_REF (10 min) → ANALYSIS (60 min) → VISUAL_GUIDE (15 min) → SETUP 1,12 (20 min)
```

### Step 3: Reference as Needed

After initial reading, use **LESS_SELECTOR_QUICK_REFERENCE.md** and **DOCUMENTATION_INDEX.md** for ongoing lookup.

---

## 📖 File Descriptions

### 🗺️ DOCUMENTATION_INDEX.md
**Your navigation hub for all documentation**

Use this to:
- Find which document answers your question
- See recommended reading order by experience level
- Understand topic coverage across documents
- Get quick answers to FAQs
- Plan your learning path

**Key Sections:**
- "Quick Start by Use Case" (6 scenarios with paths)
- "Information by Topic" (topics with document references)
- "Finding Answers to Common Questions" (Q&A index)
- "Reading Paths by Experience Level" (beginner/intermediate/advanced)

---

### 📄 LESS_SELECTOR_README.md
**Quick overview for first-time readers**

This is your **first stop** if you're new to LESS.

Includes:
- What is LESS and why it matters
- High-level 10-step process summary
- Key concepts (gradient, projection, similarity)
- When to use LESS vs other selectors
- Quick configuration example
- Pointer to detailed documentation

Read this first (~5-10 min), then pick a detailed doc.

---

### 📊 LESS_SELECTOR_VISUAL_GUIDE.md
**Flowcharts and diagrams for visual learners**

Best for understanding:
- Overall architecture and data flow
- The three phases of LESS selection
- Multi-GPU synchronization patterns
- Training loop timeline and triggers
- Directory structure during execution

Includes:
- 5+ ASCII flowcharts
- Execution timeline diagrams
- Multi-GPU synchronization visualization
- Selection pipeline breakdown
- Training sequence examples

Use this to **get the big picture** without diving into code.

---

### 🔍 LESS_SELECTOR_QUICK_REFERENCE.md
**One-page lookups for known readers**

Use this for:
- Parameter definitions (quick one-liners)
- Code location index (file:line format)
- Typical values for different scenarios
- Memory requirements calculations
- Troubleshooting checklist
- Common configurations

This is your **bookmark file** - refer to it constantly while working with LESS.

Sections:
1. Parameters (one-liner definitions)
2. Key Values and Meanings
3. Common Configurations
4. Code Locations
5. Function Signatures
6. Running Commands
7. Troubleshooting
8. Memory Requirements

---

### 🧠 LESS_SELECTOR_ANALYSIS.md
**Line-by-line implementation walkthrough**

For developers who need to:
- Understand exactly how code works
- Modify or extend LESS functionality
- Implement custom selectors
- Debug complex issues
- Understand distributed training patterns

Includes:
- Complete function-by-function walkthrough
- Code snippets with line numbers
- Explanation of logic and flow
- Integration with rest of DataFlex
- Distributed training patterns
- Memory and performance characteristics

Use this for **deep technical understanding**.

---

### ⚙️ LESS_SETUP_AND_CONFIGURATION.md
**Complete setup and operations guide**

For users who need to:
- Configure LESS for experiments
- Run LESS training on their hardware
- Debug training issues
- Understand all configuration options
- Estimate memory requirements
- Monitor experiments
- Add custom datasets
- Customize LESS behavior

Includes 13 major sections:
1. LESS Selector Configuration
2. LESS Selection Pipeline
3. Distributed Training Mechanics
4. Gradient Processing Details
5. Scoring and Selection
6. Integration with Training Loop
7. Data Configuration
8. Running LESS Experiments
9. Monitoring and Debugging
10. Memory Considerations
11. Comparison with Other Selectors
12. Common Issues and Solutions
13. Advanced Customization

Use this for **practical setup and operation**.

---

## 💡 Common Use Cases and Recommended Files

### "How do I understand what LESS does?"
1. Read LESS_SELECTOR_README.md (5 min)
2. Look at LESS_SELECTOR_VISUAL_GUIDE.md sections 1-2 (10 min)
3. Read LESS_SETUP_AND_CONFIGURATION.md sections 1-3 (15 min)

### "How do I run a LESS experiment?"
1. Skim LESS_SETUP_AND_CONFIGURATION.md sections 1-3 (10 min)
2. Copy config from section 1.2 (2 min)
3. Run command from section 8.1 (instant)
4. Monitor per section 9 (ongoing)

### "What does parameter X do?"
1. Search LESS_SELECTOR_QUICK_REFERENCE.md section 2 (1 min)
2. Cross-reference LESS_SETUP_AND_CONFIGURATION.md section 1 for details (2 min)

### "How do I modify LESS for [specific need]?"
1. Read LESS_SELECTOR_ANALYSIS.md relevant section (15-30 min)
2. Reference LESS_SELECTOR_QUICK_REFERENCE.md section 4 for code locations (2 min)
3. Check LESS_SETUP_AND_CONFIGURATION.md section 12 for customization patterns (10 min)

### "Something's wrong with my experiment"
1. Check LESS_SETUP_AND_CONFIGURATION.md section 11 (5 min)
2. If not found, read LESS_SELECTOR_QUICK_REFERENCE.md section 7 (5 min)
3. Deep dive into LESS_SELECTOR_ANALYSIS.md if needed (15-45 min)

### "How much memory will I need?"
1. Read LESS_SETUP_AND_CONFIGURATION.md section 9.3 (5 min)
2. Use LESS_SELECTOR_QUICK_REFERENCE.md section 8 calculator (2 min)

### "I want to understand the distributed training implementation"
1. Read LESS_SELECTOR_VISUAL_GUIDE.md section 3 (10 min)
2. Study LESS_SELECTOR_ANALYSIS.md sections 3-5 (20 min)
3. Reference LESS_SETUP_AND_CONFIGURATION.md section 3 (5 min)

---

## 📊 Documentation Coverage

Each document covers different aspects at different depths:

```
Understanding Architecture:    VISUAL_GUIDE ████████ ANALYSIS ███ SETUP ██ QUICK_REF █
Configuration Details:         SETUP ███████ QUICK_REF ███████ ANALYSIS ██ VISUAL █
Implementation Details:        ANALYSIS ████████ SETUP ███ QUICK_REF ██ VISUAL █
Usage and Operations:          SETUP ███████ QUICK_REF ██████ ANALYSIS ██ VISUAL █
Memory/Performance Planning:   SETUP ██████ QUICK_REF ██████ ANALYSIS ██ VISUAL █
Code Location Reference:       QUICK_REF ███████ ANALYSIS ████ SETUP █ VISUAL █
Visual Diagrams:               VISUAL_GUIDE ████████ SETUP ██ QUICK_REF █ ANALYSIS █
```

---

## 🎓 Learning Path Examples

### For ML Researchers
Goal: Understand LESS algorithm and modify for experiments

Path:
1. LESS_SELECTOR_README.md (quick overview)
2. LESS_SELECTOR_VISUAL_GUIDE.md (big picture)
3. LESS_SETUP_AND_CONFIGURATION.md sections 4-5 (gradient/scoring details)
4. LESS_SELECTOR_ANALYSIS.md (implementation details)
5. Bookmark LESS_SELECTOR_QUICK_REFERENCE.md

Time: ~90 min

### For ML Engineers
Goal: Deploy LESS for production training

Path:
1. LESS_SELECTOR_README.md (quick overview)
2. LESS_SETUP_AND_CONFIGURATION.md sections 1-3, 8-9 (config and running)
3. LESS_SETUP_AND_CONFIGURATION.md section 11 (troubleshooting)
4. LESS_SELECTOR_VISUAL_GUIDE.md section 3 (distributed training)
5. Bookmark LESS_SELECTOR_QUICK_REFERENCE.md

Time: ~60 min

### For ML Infrastructure
Goal: Integrate LESS into training pipeline

Path:
1. LESS_SELECTOR_VISUAL_GUIDE.md (architecture)
2. LESS_SELECTOR_ANALYSIS.md (implementation)
3. LESS_SETUP_AND_CONFIGURATION.md sections 3, 6-7 (distributed/data integration)
4. LESS_SELECTOR_QUICK_REFERENCE.md section 4 (code locations)

Time: ~120 min

### For Curious Learners
Goal: Understand how data selection works

Path:
1. LESS_SELECTOR_README.md (overview)
2. LESS_SELECTOR_VISUAL_GUIDE.md (flow and process)
3. LESS_SETUP_AND_CONFIGURATION.md sections 2, 5 (pipeline and scoring)
4. LESS_SELECTOR_QUICK_REFERENCE.md (keep for reference)

Time: ~45 min

---

## 🔗 Quick Navigation

### From DOCUMENTATION_INDEX.md
- "Quick Start by Use Case" (section 2) - Match your scenario to recommended files
- "Information by Topic" (section 3) - Find what covers your topic
- "Finding Answers to Common Questions" (section 4) - Get quick Q&A index

### From LESS_SELECTOR_QUICK_REFERENCE.md
- Section 4 "Code Locations" - Find where something is implemented
- Section 1 "Parameters" - Define what each parameter does
- Section 7 "Troubleshooting" - Find solutions to common problems

### From LESS_SELECTOR_VISUAL_GUIDE.md
- "Selection Pipeline" - Understand the 3-phase process
- "Multi-GPU Synchronization" - Understand distributed training
- "Directory Structure" - See what files are created where

### From LESS_SETUP_AND_CONFIGURATION.md
- Section 1 "Configuration" - Understand all config parameters
- Section 8 "Running Experiments" - How to start training
- Section 11 "Common Issues" - Troubleshoot problems

---

## 🛠️ Quick Command Reference

### Run LESS Training (Basic)
```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
  dataflex-cli train examples/train_lora/selectors/less.yaml
```

See LESS_SETUP_AND_CONFIGURATION.md section 8 for more options.

### Check LESS Configuration
```bash
cat src/dataflex/configs/components.yaml | grep -A 6 "^  less:"
```

### View Cached Selections
```bash
cat ../dataflex_saves/less_output/step_0.json | jq .
```

### Check Gradient Files
```bash
ls -lh ../dataflex_saves/less_output/train/0/
```

See LESS_SETUP_AND_CONFIGURATION.md section 9.2 for more debugging commands.

---

## ❓ FAQ

### Q: Which document should I read first?
**A:** LESS_SELECTOR_README.md (5-10 min), then choose one of the detailed docs based on your goal.

### Q: I'm in a hurry. What's the minimum I need to know?
**A:** Read LESS_SELECTOR_README.md, then LESS_SELECTOR_VISUAL_GUIDE.md section "Selection Pipeline", then LESS_SETUP_AND_CONFIGURATION.md section 1. (30 min total)

### Q: Can I just copy a config and run it?
**A:** Yes! Copy examples/train_lora/selectors/less.yaml and run. See LESS_SETUP_AND_CONFIGURATION.md section 8.1 for the command.

### Q: How do I find X in the code?
**A:** Use LESS_SELECTOR_QUICK_REFERENCE.md section 4 "Code Locations" to find the file and line number.

### Q: My experiment is failing. Where do I look?
**A:** Check LESS_SETUP_AND_CONFIGURATION.md section 11 "Common Issues and Solutions" first. If not found, see LESS_SELECTOR_QUICK_REFERENCE.md section 7 "Troubleshooting Checklist".

### Q: Can I modify LESS?
**A:** Yes! See LESS_SETUP_AND_CONFIGURATION.md section 12 "Advanced Customization" for examples, and LESS_SELECTOR_ANALYSIS.md for implementation details.

### Q: How much memory will I need?
**A:** See LESS_SETUP_AND_CONFIGURATION.md section 9.3 "Memory Considerations" with formulas and examples.

### Q: Why are there 6 documentation files?
**A:** Each serves a different purpose:
- INDEX: Navigation hub
- README: Quick overview
- VISUAL: Big picture thinking
- QUICK_REF: Fast lookups
- ANALYSIS: Deep understanding
- SETUP: Practical operations

This gives you exactly what you need at any given moment, without forcing you to read irrelevant content.

---

## 📋 Checklist: What You Should Know After Reading

After reading the appropriate documentation for your role:

### After LESS_SELECTOR_README.md
- [ ] I can explain what LESS does in 2 sentences
- [ ] I know when to use LESS vs other selectors
- [ ] I know the 10 basic steps of LESS selection

### After LESS_SELECTOR_VISUAL_GUIDE.md
- [ ] I can draw the selection pipeline from memory
- [ ] I understand what happens on each GPU
- [ ] I know how main process coordinates selections

### After LESS_SELECTOR_QUICK_REFERENCE.md
- [ ] I can look up any parameter definition
- [ ] I know where to find functions in code
- [ ] I can estimate memory for my experiment

### After LESS_SELECTOR_ANALYSIS.md
- [ ] I understand every major function in detail
- [ ] I could modify or extend LESS myself
- [ ] I understand distributed training patterns

### After LESS_SETUP_AND_CONFIGURATION.md
- [ ] I can configure LESS for my use case
- [ ] I know how to run experiments
- [ ] I can debug common issues
- [ ] I understand all config parameters

---

## 🎯 Final Notes

These 6 documentation files represent **comprehensive coverage** of the LESS selector:

- **~100KB** of detailed information
- **100+ code snippets** with line references
- **10+ ASCII diagrams** and flowcharts
- **12+ common issues** with solutions
- **8 configuration examples**
- **Cross-references** between documents

They are designed to be **read selectively** - start with the file that matches your current goal, then cross-reference with others as needed.

---

## 📞 Support Resources

### If you can't find an answer:
1. Check DOCUMENTATION_INDEX.md section "Finding Answers to Common Questions"
2. Use Ctrl+F to search relevant documents
3. Check LESS_SELECTOR_QUICK_REFERENCE.md for code locations
4. Refer to source files:
   - `src/dataflex/train/selector/less_selector.py` (implementation)
   - `examples/train_lora/selectors/less.yaml` (config example)
   - `src/dataflex/configs/components.yaml` (component templates)

---

**Last Updated:** 2026-05-13
**Documentation Version:** 1.0
**LESS Implementation Version:** Latest in DataFlex

Happy learning! 🚀
