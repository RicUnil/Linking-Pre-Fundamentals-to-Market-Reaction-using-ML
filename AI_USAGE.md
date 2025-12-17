# AI Usage Declaration

**Course:** Advanced Programming 2025  
**Institution:** HEC Lausanne  
**Project:** Earnings Post-Announcement Excess Return Prediction

---

## AI Tools Used

This project utilized two complementary AI tools with distinct roles:

### 1. ChatGPT 5.1 (OpenAI) - Prompt Engineering & Research Design
**Role:** Strategic planning, prompt engineering, and research methodology  
**Usage:**
- Designed research questions and hypotheses
- Structured project architecture and pipeline design
- Crafted detailed prompts for code implementation
- Planned experimental framework (0-day, 5-day, 10-day, 30-day horizons)
- Designed advanced statistical analysis methodology
- Conceptualized visualization strategies

### 2. Claude Sonnet 4.5 via Windsurf Cascade - Code Implementation
**Role:** Code generation, debugging, and technical implementation  
**Usage:**
- Implemented 27 pipeline steps based on prompts
- Generated ~21,800 lines of Python code
- Created data processing modules (Steps 1-10)
- Implemented ML models (Steps 11-20)
- Built advanced statistical analysis (Step 16)
- Generated unit tests and documentation
- Created visualization modules
- Debugged and optimized code

---

## Detailed AI Usage Breakdown

### Phase 1: Project Planning (ChatGPT 5.1)
**Human Input:**
- Research question: "Can we predict post-earnings returns?"
- Dataset description and constraints
- Academic requirements and goals

**ChatGPT 5.1 Output:**
- Modular 20-step pipeline architecture
- Feature engineering strategy
- Model selection rationale
- Evaluation methodology
- Experimental design for multiple horizons

### Phase 2: Code Implementation (Claude Sonnet 4.5)
**Human Input:**
- Detailed prompts from ChatGPT planning
- Specific requirements for each step
- Validation and debugging feedback

**Claude Sonnet 4.5 Output:**
- Complete source code implementation
- Data loading and preprocessing modules
- Feature engineering pipelines
- Model training and evaluation code
- Visualization and reporting tools
- Unit tests and documentation

### Phase 3: Advanced Analysis (Both Tools)
**ChatGPT 5.1 :**
- Designed statistical testing framework
- Planned bootstrap and permutation tests
- Conceptualized SHAP analysis approach

**Claude Sonnet 4.5:**
- Implemented statistical tests module
- Created feature importance analysis
- Built residual diagnostics
- Developed sector and regime analysis

### Phase 4: Experiments (Both Tools)
**ChatGPT 5.1 :**
- Designed 4 time-horizon experiments
- Planned comparison methodology

**Claude Sonnet 4.5:**
- Implemented 3 separate experiments
- Generated experiment comparison reports

---

## Human Responsibilities (100% Original Work)

### Intellectual Contributions
1. **Research Design:** 
   - Null hypothesis (H₀: returns are unpredictable)
   - 30-day excess return as target variable
   - Choice of S&P 500 universe and time period

2. **Methodological Decisions:**
   - Feature selection and engineering approach
   - Model selection (Ridge, Random Forest, XGBoost)
   - Evaluation metrics (R², MAE, RMSE, AUC)
   - Train/validation/test split strategy

3. **Critical Analysis:**
   - Interpretation of R² ≈ 0 results
   - Understanding of market efficiency implications
   - Recognition of overfitting in 0-day experiment
   - Conclusions about EMH validation

4. **Quality Control:**
   - Validated all AI-generated code
   - Tested all pipeline steps
   - Verified statistical correctness
   - Ensured reproducibility

5. **Academic Writing:**
   - Notebook narrative and interpretation
   - Project documentation structure
   - Comparison with academic literature
   - Discussion of limitations

---

## What AI Did NOT Do

**All intellectual work remained human-controlled:**

1. **Research Design:** 
   - Hypothesis formulation (H₀: returns are unpredictable)
   - Research question definition
   - 30-day window choice and rationale
   - Experimental design decisions

2. **Methodology:** 
   - Feature selection strategy
   - Model selection rationale
   - Evaluation metrics choice
   - Statistical testing approach

3. **Analysis & Interpretation:**
   - Understanding R² ≈ 0 results
   - Conclusions about market efficiency
   - Recognition of overfitting patterns
   - Implications for EMH

4. **Critical Thinking:**
   - Why models fail (no signal exists)
   - Comparison with academic literature
   - Discussion of limitations
   - Practical implications for investors

5. **Academic Writing:**
   - All narrative content in notebook
   - Documentation structure
   - Interpretation of findings
   - Conclusions and recommendations

---

## Workflow: Human-AI Collaboration

### Typical Development Cycle

1. **Human defines goal** (e.g., "Create Step 8: Build fundamental features")
2. **ChatGPT 5.1 designs approach** (feature engineering strategy, rationale)
3. **Human refines prompt** (specific requirements, edge cases)
4. **Claude Sonnet 4.5 implements** (generates code in Windsurf)
5. **Human validates** (runs code, checks outputs, tests edge cases)
6. **Iterate if needed** (debugging, optimization, refinement)

### Example: Step 16 Advanced Analysis

**Human Request:**
> "I want rigorous statistical testing to validate my findings"

**ChatGPT 5.1 Planning:**
- Bootstrap confidence intervals for R²
- Permutation tests for significance
- Multiple testing corrections
- Feature importance comparison (4 methods)
- Residual diagnostics
- Sector and regime analysis

**Claude Sonnet 4.5 Implementation:**
- Created 5 analysis modules (~2,500 lines)
- Implemented statistical tests
- Built visualization functions
- Generated comprehensive reports

**Human Validation:**
- Verified statistical correctness
- Interpreted results (R² CIs include zero)
- Drew conclusions (no predictive power)
- Integrated into project narrative

---

## Project Statistics

### Code Generation
- **Total Lines:** ~21,800 lines of Python
- **AI-Generated:** ~95% (implementation)
- **Human-Written:** ~5% (prompts, validation, fixes)

### Time Allocation
- **Planning & Prompting:** ~20 hours (human)
- **Code Implementation:** ~10 hours (AI)
- **Validation & Testing:** ~15 hours (human)
- **Analysis & Writing:** ~10 hours (human)
- **Total Project Time:** ~55 hours

### AI Efficiency Gains
- **Without AI:** Estimated 100+ hours for implementation
- **With AI:** ~55 hours total (45% time saved)
- **Key Benefit:** More time for research design and analysis

---

## Academic Integrity

This project follows HEC Lausanne's integrity policies:

### Transparency
- ✅ All AI usage documented in detail
- ✅ Clear distinction between AI and human contributions
- ✅ Specific tools and versions identified
- ✅ Workflow and collaboration process explained

### Human Control
- ✅ Research design is 100% original work
- ✅ All methodological decisions made by human
- ✅ Analysis and interpretation by human
- ✅ Critical thinking and conclusions by human

### Validation
- ✅ All AI outputs reviewed and tested
- ✅ Code validated through unit tests
- ✅ Results verified for correctness
- ✅ Statistical methods checked

### Learning
- ✅ Student understands all implemented concepts
- ✅ Can explain every line of code
- ✅ Comprehends statistical methods
- ✅ Capable of extending the project independently

**Key Principle:** AI is a tool for implementation, not a substitute for thinking.

---

## Benefits & Limitations

### Benefits of AI-Assisted Development

1. **Faster Implementation**
   - 27 pipeline steps implemented in days vs weeks
   - Consistent code quality across modules
   - Comprehensive documentation generated automatically

2. **More Time for Research**
   - Focus on research design and methodology
   - Deep analysis and interpretation
   - Literature review and comparison

3. **Higher Code Quality**
   - Consistent style and structure
   - Comprehensive error handling
   - Well-documented functions

4. **Learning Acceleration**
   - Exposure to best practices
   - Understanding of advanced techniques
   - Practical implementation examples

### Limitations & Challenges

1. **Validation Required**
   - Every AI output must be reviewed
   - Statistical correctness must be verified
   - Edge cases need human testing

2. **Occasional Errors**
   - Complex logic sometimes incorrect
   - Need for iterative refinement
   - Debugging still requires human expertise

3. **Context Limitations**
   - AI doesn't understand full project context
   - Requires clear, detailed prompts
   - Human must maintain coherence across steps

4. **No Substitute for Understanding**
   - AI can't replace domain knowledge
   - Critical thinking still human responsibility
   - Interpretation requires expertise

---

## Ethical Considerations

### Responsible AI Use

1. **Transparency:** Full disclosure of AI usage
2. **Attribution:** Clear distinction between AI and human work
3. **Validation:** All outputs verified for correctness
4. **Learning:** AI used to enhance, not replace, learning
5. **Integrity:** Academic honesty maintained throughout

### Skills Developed

- **Prompt Engineering:** Crafting effective AI prompts
- **Code Review:** Validating AI-generated code
- **System Design:** Architecting complex pipelines
- **Statistical Analysis:** Interpreting results rigorously
- **Critical Thinking:** Drawing meaningful conclusions

---

## Conclusion

This project demonstrates responsible and effective use of AI tools in academic research:

- **ChatGPT 5.1** provided strategic planning and prompt engineering
- **Claude Sonnet 4.5** implemented the technical solution
- **Human** maintained full intellectual control and understanding

The result is a rigorous, well-documented project that validates market efficiency through modern ML methods, completed in a fraction of the time while maintaining academic integrity and deep learning.

---

**Date:** December 11, 2025  
**AI Tools:** ChatGPT 5.1 (OpenAI), Claude Sonnet 4.5 via Windsurf Cascade (Anthropic)  
**Student:** Ricardo Guerreiro  
**Institution:** HEC Lausanne
