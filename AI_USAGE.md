# AI Usage Documentation

**Course:** Advanced Programming 2025  
**Student:** Ricardo Contente Guerreiro  
**Institution:** HEC Lausanne

---

## AI Tools Used

This project used AI tools as learning aids and productivity enhancers:

- **ChatGPT 5.1 (OpenAI)** - Planning and prompt engineering
- **Claude Sonnet 4.5 via Windsurf Cascade (Anthropic)** - Code implementation and debugging

---

## ✅ Acceptable Uses (How I Used AI)

### 1. Debugging Help
- **What:** Fixed import errors, data loading issues, and date column detection bugs
- **Example:** Resolved `ModuleNotFoundError` in step 16 by correcting import paths
- **Understanding:** I understand the Python module system and can explain why the fix worked

### 2. Learning New Libraries
- **What:** Used AI to learn SHAP for feature importance and statistical testing libraries
- **Example:** Implemented bootstrap confidence intervals and permutation tests
- **Understanding:** I comprehend the statistical methods and can interpret the results

### 3. Code Review Suggestions
- **What:** AI reviewed code for best practices, error handling, and optimization
- **Example:** Improved data loading to preserve datetime index in parquet files
- **Understanding:** I validated all suggestions and understand the improvements

### 4. Documentation Writing
- **What:** Generated docstrings and README sections following NumPy style
- **Example:** Created comprehensive function documentation with type hints
- **Understanding:** I reviewed and edited all documentation for accuracy

---

## ❌ What I Did NOT Do

- ❌ **Did not** submit code I don't understand
- ❌ **Did not** have AI write the entire project without my involvement
- ❌ **Did not** blindly accept AI outputs without validation

---

## Human Contributions (My Original Work)

### Research Design (100% Human)
- Formulated research question: "Can we predict post-earnings returns?"
- Designed 30-day excess return target variable
- Selected S&P 500 universe and 2015-2024 time period
- Chose evaluation metrics (R², MAE, RMSE, AUC)

### Critical Analysis (100% Human)
- Interpreted R² ≈ 0 results as evidence for market efficiency
- Recognized overfitting in 0-day experiment (val R² = 0.38, test R² = -0.45)
- Drew conclusions about Efficient Market Hypothesis
- Compared findings with academic literature (Ball & Brown 1968, Fama 1970)

### Validation & Testing (100% Human)
- Tested all 22 pipeline steps individually
- Verified statistical correctness of hypothesis tests
- Validated model outputs and predictions
- Ensured reproducibility with `random_state=42`

### Project Management (100% Human)
- Structured 22-step modular pipeline
- Designed 5 robustness experiments
- Created project timeline and milestones
- Maintained Git version control

---

## Workflow: Human-AI Collaboration

**Typical cycle for each pipeline step:**

1. **I define the goal** (e.g., "Build fundamental features from earnings data")
2. **AI implements** (generates code based on my specifications)
3. **I validate** (run code, check outputs, test edge cases)
4. **I debug** (fix errors, optimize, refine with AI assistance)
5. **I understand** (can explain every line and design decision)

**Example: Step 16 Advanced Analysis**
- **My request:** "Implement bootstrap CIs and permutation tests for R²"
- **AI generated:** Statistical testing module (~500 lines)
- **I validated:** Verified bootstrap resampling logic, checked p-values
- **I interpreted:** Concluded that R² CIs include zero → no predictive power

---

## Learning Outcomes

Through this AI-assisted project, I developed:

1. **Technical Skills:**
   - Python ML pipeline design (pandas, scikit-learn, XGBoost)
   - Statistical hypothesis testing (bootstrap, permutation tests)
   - Feature engineering for financial data
   - Model evaluation and validation

2. **Critical Thinking:**
   - Interpreting negative results (R² ≈ 0)
   - Understanding market efficiency implications
   - Recognizing overfitting patterns
   - Drawing evidence-based conclusions

3. **AI Collaboration Skills:**
   - Writing effective prompts
   - Validating AI outputs critically
   - Debugging AI-generated code
   - Maintaining intellectual control

**Key Principle:** I used AI as a productivity tool, not a thinking substitute. All intellectual work, analysis, and understanding are my own.

---

## Academic Integrity Statement

I certify that:

- ✅ I understand every line of code in this project
- ✅ I can explain all methodological decisions
- ✅ I independently analyzed and interpreted all results
- ✅ All AI usage is transparently documented
- ✅ This project represents my learning and understanding

AI accelerated implementation, but **I maintained full intellectual control and comprehension** throughout.

---

**Date:** December 21, 2025  
**Ricardo Guerreiro | MSc Finance**
