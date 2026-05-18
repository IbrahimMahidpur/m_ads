"""
Code Execution Agent — hardened sandbox with resource limits.
FIX: working_dir now includes session_id for session isolation.
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
import shutil
import concurrent.futures
from pathlib import Path
from typing import Optional

import httpx

from multimodal_ds.config import CODER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.observability import agent_span, get_session_tracker

logger = logging.getLogger(__name__)

_CPU_SECONDS    = int(os.getenv("SANDBOX_CPU_SECONDS",  "60"))
_MEM_MB         = int(os.getenv("SANDBOX_MEM_MB",       "512"))
_STDOUT_CHARS   = int(os.getenv("SANDBOX_STDOUT_CHARS", "8000"))
_PROC_TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S",    "300"))

SYSTEM_PROMPT = """You are a senior data scientist. You write precise, self‑contained Python.

MANDATORY RULES — follow every one, no exceptions:
1. First line of code: print(df.columns.tolist()) and print(df.shape)
2. Use ONLY column names confirmed by step 1 — never guess column names
3. Print descriptive stats: df.describe(), value_counts for every categorical column
4. ALWAYS follow this EXACT preprocessing sequence before any model.fit():
   Step A — Identify target and drop useless columns:
       target_col = 'Exited'  # or whatever the target is
       drop_cols = [c for c in df.columns if df[c].nunique() == df.shape[0]]  # IDs
       drop_cols += ['RowNumber', 'CustomerId', 'Surname']  # known non-predictive
       drop_cols = [c for c in drop_cols if c in df.columns]
       df = df.drop(columns=drop_cols, errors='ignore')

   Step B — Encode ALL categoricals with pd.get_dummies (never leave object cols):
       df = pd.get_dummies(df, drop_first=True)

   Step C — Separate features and target AFTER encoding:
       y = df[target_col].astype(int)
       X = df.drop(columns=[target_col])

   Step D — Fill any remaining NaN:
       X = X.fillna(X.median())

   Step E — Verify no object columns (print, do NOT assert):
       obj_cols = X.select_dtypes(include=['object']).columns.tolist()
       if obj_cols:
           print(f'WARNING: dropping remaining object cols: {obj_cols}')
           X = X.drop(columns=obj_cols)
       print(f'Final X shape: {X.shape}, dtypes OK')

   Step F — Split:
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42, stratify=y)

   NEVER use assert statements — use if/print/drop instead.
   NEVER call model.fit() before completing Steps A-F.
5. ALWAYS set matplotlib backend as the ABSOLUTE FIRST THREE LINES of the entire script,
   before ANY other import including pandas, numpy, sklearn:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   These 3 lines must appear at line 1, 2, 3 of the file. No exceptions.
   Never import matplotlib.pyplot before calling matplotlib.use('Agg').
6. Save ALL plots: plt.savefig('filename.png', dpi=100, bbox_inches='tight',
   facecolor='white', edgecolor='none')
   Then IMMEDIATELY call plt.close('all') after every savefig call.
   Never reuse a figure across multiple plots.
7. Never call plt.show() under any circumstances.
8. When creating subplot grids with plt.subplots(rows, cols):
   - ALWAYS flatten axes immediately after creation:
     fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
     axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
   - ALWAYS calculate nrows and ncols from actual data AFTER loading:
     n_cols = len(columns_to_plot)
     ncols = min(3, n_cols)
     nrows = (n_cols + ncols - 1) // ncols  # ceiling division
   - ALWAYS iterate with enumerate and check bounds:
     for i, col in enumerate(columns_to_plot):
         if i >= len(axes): break
         axes[i].hist(df[col].dropna(), bins=30)
   - ALWAYS hide unused axes:
     for j in range(i+1, len(axes)): axes[j].set_visible(False)
   - ALWAYS call fig.tight_layout() before savefig
   - NEVER use axes[i] without first calling axes = axes.flatten()
9. Save trained models: joblib.dump(model, 'model.pkl')
10. End with a FINDINGS block: print('=== FINDINGS ===') then 3‑5 quantitative sentences
11. NEVER evaluate a model on training data. Always use held-out test set.
   If cross-validation: use cross_val_score with cv=5 on training data only.
12. Keep ALL string literals on a single line. Never split a string literal across lines using implicit continuation. For long titles use short versions: ax.set_title('H2: Products') not ax.set_title('H2: NumOfProducts > 2 has significantly higher churn rates than customers with 1 or 2 products')
13. NEVER import seaborn before setting matplotlib backend. Import order MUST be:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns  # only after the above
14. NEVER use assert statements in data science code. Assertions crash the script.
    Instead use: if condition: print('warning'); X = X.drop(...)
15. For Churn_Modelling.csv specifically, ALWAYS drop these columns before modeling:
    ['RowNumber', 'CustomerId', 'Surname']
    ALWAYS encode with pd.get_dummies(df, drop_first=True) BEFORE separating X and y.
    The target column is 'Exited'.
16. ALWAYS convert boolean columns to int immediately after pd.get_dummies() or pd.read_csv():
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    This prevents numpy.histogram RuntimeWarning and crashes when plotting
Output only valid Python code inside ```python ... ``` fences. No commentary outside the fences."""


def _sandbox_preexec() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        mem_bytes = _MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass


def _is_only_warnings(stderr: str) -> bool:
    """Check if stderr contains only warnings (e.g. deprecation/future warnings) and no actual exception traceback or fatal error."""
    if not stderr.strip():
        return True
    stderr_lower = stderr.lower()
    # If a traceback is present, it's a real crash
    if "traceback" in stderr_lower or "stack traceback" in stderr_lower:
        return False
    # Typical error keywords indicating execution failed
    error_keywords = [
        "error:", "exception:", "failed:", "exit status",
        "nameerror", "syntaxerror", "typeerror", "valueerror",
        "keyerror", "indexerror", "attributeerror", "importerror",
        "modulenotfounderror", "zerodivisionerror", "runtimeerror"
    ]
    if any(kw in stderr_lower for kw in error_keywords):
        return False
    # If the stderr has warnings, but no traceback or standard errors, treat as warnings
    return "warning" in stderr_lower


class CodeExecutionAgent:
    AGENT_NAME = "code_execution_agent"

    def __init__(self, working_dir: Optional[str] = None, session_id: str = "default"):
        # FIX: include session_id in working_dir for session isolation
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.memory = AgentMemory()
        self._tracker = get_session_tracker(session_id)

    def _enforce_matplotlib_backend(self, code: str) -> str:
        """Move matplotlib backend setup to lines 1‑3 if not already there."""
        import re
        backend_block = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
        )
        # Check first three lines for correct placement
        first_three = "\n".join(code.splitlines()[:3])
        if "matplotlib.use('Agg')" in first_three:
            return code
        # Remove existing matplotlib imports
        code = re.sub(r'^import matplotlib\n', '', code, flags=re.MULTILINE)
        code = re.sub(r"^matplotlib\.use\(['\"]Agg['\"]\)\n", '', code, flags=re.MULTILINE)
        code = re.sub(r'^import matplotlib\.pyplot as plt\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^from matplotlib import.*\n', '', code, flags=re.MULTILINE)
        return backend_block + code
    
    def _fix_common_code_errors(self, code: str) -> str:
        """
        Apply only minimal, safe, mechanical fixes that are 100% reliable.
        All semantic/API fixes are handled by the LLM in _generate_fix().
        """
        import re

        # Only apply fixes that are syntactically mechanical and never wrong:

        # 1. Remove duplicate matplotlib backend declarations
        code = re.sub(
            r'(import matplotlib\s*\n\s*matplotlib\.use\([\'"]Agg[\'"]\)\s*\n)',
            '',
            code,
            count=max(0, code.count("matplotlib.use('Agg')") - 1),
        )

        # 2. Flatten axes after plt.subplots — always safe
        code = re.sub(
            r'(fig\s*,\s*axes\s*=\s*plt\.subplots\([^)]+\))',
            r'\1\naxes = axes.flatten() if hasattr(axes, "flatten") else [axes]',
            code,
        )

        # Fix: close unclosed parentheses on lines containing figsize=
        # LLMs frequently generate: plt.subplots(r, c, figsize=(w, h)
        # with a missing closing paren for plt.subplots
        # Fix: plt.subplots() calls with figsize often have a missing closing paren
        # because the LLM splits the call across lines:
        #   fig, axes = plt.subplots(nrows, ncols,
        #                            figsize=(4 * ncols, 4 * nrows)   ← missing )
        # Strategy: find every plt.subplots( call, collect its full span
        # across continuation lines, recount parens, append missing closers.
        import re as _re2
        lines = code.splitlines()
        i = 0
        fixed_lines = []
        while i < len(lines):
            line = lines[i]
            if 'plt.subplots(' in line:
                # Collect this line and any indented continuation lines
                block = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # Continuation if indented more than zero OR starts with figsize/spaces
                    stripped = next_line.strip()
                    if stripped and (next_line.startswith('    ') or next_line.startswith('\t') or
                                     stripped.startswith('figsize') or stripped.startswith(')')):
                        # Check if the block is still unbalanced
                        block_text = '\n'.join(block)
                        if block_text.count('(') > block_text.count(')'):
                            block.append(next_line)
                            j += 1
                            continue
                    break

                # Now recount parens across the whole block
                block_text = '\n'.join(block)
                open_count  = block_text.count('(')
                close_count = block_text.count(')')
                if open_count > close_count:
                    # Append missing closing parens to the last line of the block
                    missing = open_count - close_count
                    block[-1] = block[-1].rstrip() + ')' * missing
                    logger.info(
                        f"[CodeAgent] Auto-closed {missing} paren(s) in plt.subplots block"
                    )

                fixed_lines.extend(block)
                i = j
            else:
                fixed_lines.append(line)
                i += 1

        code = '\n'.join(fixed_lines)
        return code
    
    # ADD this new method to CodeExecutionAgent class, after _fix_common_code_errors:

    def _validate_and_fix_syntax(self, code: str) -> tuple[str, bool]:
        """Validate Python syntax. If broken, use the LLM to fix it."""
        import ast

        def _check(c: str) -> tuple[bool, str]:
            try:
                ast.parse(c)
                return True, ""
            except SyntaxError as e:
                return False, f"Line {e.lineno}: {e.msg}\n{e.text}"

        valid, error = _check(code)
        if valid:
            return code, True

        logger.warning(f"[CodeAgent] Syntax error: {error} — asking LLM to fix")

        fix_prompt = f"""This Python code has a syntax error. Fix ONLY the syntax error and return the complete corrected script.

Syntax error:
{error}

Code with error:
```python
{code}
```

Return the complete fixed Python script in ```python ... ``` fences.
Do not change logic, only fix the syntax."""

        try:
            from multimodal_ds.core.llm_client import chat_with_fallback
            fixed_raw = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "You are a Python syntax fixer. Fix syntax errors only. Return complete code in ```python``` fences."},
                    {"role": "user", "content": fix_prompt},
                ],
                max_tokens=8000,
                temperature=0.0,
            )
            fixed_code = self._extract_code(fixed_raw)
            if fixed_code:
                valid, error = _check(fixed_code)
                if valid:
                    logger.info("[CodeAgent] LLM self-healed the syntax error")
                    return fixed_code, True
                else:
                    logger.warning(f"[CodeAgent] LLM fix still has syntax error: {error}")
        except Exception as e:
            logger.warning(f"[CodeAgent] LLM syntax fix failed: {e}")

        return code, False

        logger.warning(f"[CodeAgent] Syntax error detected: {error} — attempting auto-repair")

        # Auto-repair pass 1: Fix unclosed parentheses/brackets by counting
        def _fix_unclosed(c: str) -> str:
            opens  = {'(': ')', '[': ']', '{': '}'}
            closes = {v: k for k, v in opens.items()}
            stack  = []
            for ch in c:
                if ch in opens:
                    stack.append(opens[ch])
                elif ch in closes:
                    if stack and stack[-1] == ch:
                        stack.pop()
            # Append missing closing chars
            if stack:
                c = c.rstrip() + ''.join(reversed(stack))
                logger.info(f"[CodeAgent] Auto-closed: {''.join(reversed(stack))}")
            return c

        # Auto-repair pass 2: Fix common LLM multi-line syntax breaks
        def _fix_broken_lines(c: str) -> str:
            lines = c.splitlines()
            fixed = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Detect line ending with an open paren/bracket but no content after
                stripped = line.rstrip()
                if stripped.endswith('(') or stripped.endswith(','):
                    # Check if next line looks like a continuation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # If next line starts with a value/identifier, merge
                        if next_line and not next_line.startswith('#'):
                            fixed.append(line.rstrip() + ' ' + next_line)
                            i += 2
                            continue
                fixed.append(line)
                i += 1
            return '\n'.join(fixed)

        # Auto-repair pass 3: Fix figsize tuple specifically (common LLM error)
        def _fix_figsize(c: str) -> str:
            lines = c.splitlines()
            fixed_lines = []
            for line in lines:
                stripped = line.rstrip()
                # Count parens on this line
                opens  = stripped.count('(')
                closes = stripped.count(')')
                if opens > closes:
                    # Missing closing parens — add them
                    missing = opens - closes
                    stripped = stripped + ')' * missing
                    logger.info(f"[CodeAgent] Auto-closed {missing} paren(s) on line: {stripped[:60]}")
                fixed_lines.append(stripped)
            return '\n'.join(fixed_lines)

        # Apply repairs in sequence, re-checking after each
        repairs = [_fix_figsize, _fix_broken_lines, _fix_unclosed]
        for repair_fn in repairs:
            code = repair_fn(code)
            valid, error = _check_syntax(code)
            if valid:
                logger.info(f"[CodeAgent] Syntax repaired by {repair_fn.__name__}")
                return code, True

        # If still broken, strip the offending line and everything after it
        # as a last resort — partial output is better than no output
        lines = code.splitlines()
        for i in range(len(lines) - 1, 0, -1):
            candidate = '\n'.join(lines[:i])
            valid, _ = _check_syntax(candidate)
            if valid:
                logger.warning(
                    f"[CodeAgent] Truncated code at line {i} to fix syntax — "
                    f"{len(lines) - i} lines removed"
                )
                return candidate, True

        logger.error("[CodeAgent] Syntax auto-repair failed — returning original")
        return code, False

    def _fix_filename_references(self, code: str, working_files: list[str]) -> str:
        """Detect and fix filename mismatches in generated code.

        The LLM often generates code with wrong filenames (e.g., 'Churn.csv' when
        the actual file is 'Churn_Modelling.csv'). This method:
        1. Finds file references in generated code (pd.read_csv, pd.read_excel, etc.)
        2. Checks if referenced files exist in working directory
        3. If not, finds the closest matching actual file and replaces the reference
        """
        if not working_files:
            return code

        actual_files = {Path(f).name.lower(): Path(f).name for f in working_files}
        import re

        # Patterns that reference data files in code
        file_ref_patterns = [
            r'pd\.read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'open\s*\(\s*["\']([^"\']+\.(?:csv|json|txt))["\']',
        ]

        def find_closest_match(missing_name: str) -> str | None:
            """Find the most similar actual file using fuzzy matching."""
            import os
            missing_lower = missing_name.lower()
            # Exact match (case-insensitive)
            if missing_lower in actual_files:
                return actual_files[missing_lower]

            # Extract stem (filename without extension) for smarter matching
            missing_stem = os.path.splitext(missing_name)[0].lower()
            missing_ext = os.path.splitext(missing_name)[1].lower()

            # Try to find a match by comparing stems
            for actual_lower, actual_name in actual_files.items():
                actual_stem = os.path.splitext(actual_lower)[0]
                actual_ext = os.path.splitext(actual_lower)[1]

                # Skip if extensions don't match (could be different file types)
                if actual_ext != missing_ext:
                    continue

                # Check if stems share significant overlap:
                # 1. One stem is contained in the other (e.g., "Churn" in "Churn_Modelling")
                # 2. They share the first half of characters (e.g., "custom" vs "customer")
                if (missing_stem in actual_stem or actual_stem in missing_stem or
                    (len(missing_stem) > 3 and len(actual_stem) > 3 and
                     missing_stem[:len(missing_stem)//2] in actual_stem)):
                    return actual_name

            return None

        # Find all file references and try to fix them
        for pattern in file_ref_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for ref in matches:
                ref_lower = ref.lower()
                # Check if this exact reference exists
                if ref_lower not in actual_files:
                    # Try to find a close match
                    matched = find_closest_match(ref)
                    if matched:
                        logger.info(f"[CodeAgent] Fixing filename: '{ref}' -> '{matched}'")
                        # Replace all occurrences of this wrong filename using a compiled regex
                        old_ref_pattern = re.compile(
                            r'(["\'])' + re.escape(ref) + r'(["\'])',
                            re.IGNORECASE
                        )
                        code = old_ref_pattern.sub(
                            lambda m, matched=matched: m.group(1) + matched + m.group(2),
                            code
                        )

        # If no CSV filename is referenced, add a fallback import
        if not re.search(r"\.csv", code):
            csv_files = [Path(f).name for f in working_files if f.lower().endswith('.csv')]
            if csv_files:
                actual_filename = csv_files[0]
                fallback_line = f"# Data file: {actual_filename}\ndf = pd.read_csv('{actual_filename}')"
                # Insert after matplotlib import lines if they exist (first three lines usually)
                lines = code.splitlines()
                insert_idx = 0
                # Detect the typical three matplotlib lines
                if len(lines) >= 3 and all('matplotlib' in lines[i] for i in range(3)):
                    insert_idx = 3
                else:
                    # Find first line containing 'import matplotlib'
                    for i, l in enumerate(lines):
                        if 'import matplotlib' in l:
                            insert_idx = i + 1
                            break
                # Insert the fallback line
                lines = lines[:insert_idx] + [fallback_line] + lines[insert_idx:]
                code = "\n".join(lines)
        return code

    def execute_task(self, task: dict, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        task_desc = task.get("description", str(task))
        task_name = task.get("name", "task")
        logger.info(f"[CodeAgent] Executing: {task_name}")

        with agent_span(self.AGENT_NAME, self.session_id, self._tracker) as span:
            span.set_metadata({"task_name": task_name})
            past_context = self._get_relevant_memory(task_desc)
            raw_code = self._generate_code(task_desc, data_context, past_context)
            code = self._extract_code(raw_code)
            if not code:
                # Retry code generation once with a simpler prompt before giving up.
                # The first attempt may fail if the model is still loading or
                # the context is too long. A simplified retry often succeeds.
                logger.warning("[CodeAgent] First code generation attempt returned empty — retrying with simplified prompt")
                simplified_desc = (
                    f"{task_desc}\n\n"
                    f"IMPORTANT: Respond with ONLY a Python code block inside ```python ... ``` fences. "
                    f"No explanation. No prose. Just the code."
                )
                raw_code = self._generate_code(simplified_desc, data_context[:500], "")
                code = self._extract_code(raw_code)
            if not code:
                logger.error(f"[CodeAgent] Code generation failed after retry for task: {task_desc[:100]}")
                return {
                    "success": False,
                    "error": "Code generation failed",
                    "code": "",
                    "output": "Code generation failed — LLM returned no parseable Python code.",
                    "files_created": [],
                }
            span.set_chars(input_chars=len(task_desc) + len(data_context), output_chars=len(code))
            result = self._execute_with_retry(code, task_desc, data_context, file_paths, max_retries)
            span.set_metadata({"task_name": task_name, "success": result["success"], "files_created": result["files_created"]})

        status_msg = "successfully" if result["success"] else "with errors"
        self.memory.store_analysis_step(
            step_name=task_name,
            result=f"Code executed {status_msg}.\nOutput: {result['output'][:500]}\nFiles: {result['files_created']}",
            session_id=self.session_id,
        )
        return result

    def execute(self, task_description: str, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        rag_context = self._retrieve_rag_context(task_description)
        if rag_context:
            data_context = f"Relevant document context (from ChromaDB):\n{rag_context}\n\n" + data_context
        if file_paths:
            file_list = "\n".join(f"  - {Path(fp).name}" for fp in file_paths)
            data_context = f"Available data files (use exact names):\n{file_list}\n\n{data_context}"
        task = {"name": task_description[:80], "description": task_description}
        return self.execute_task(task=task, data_context=data_context, file_paths=file_paths, max_retries=max_retries)

    def _retrieve_rag_context(self, query: str, k: int = 4) -> str:
        try:
            results = self.memory.retrieve(query, n_results=k)
            if results:
                return "\n\n".join(r["content"] for r in results if r.get("content"))
        except Exception:
            pass
        return ""

    def _generate_code(self, task_desc: str, data_context: str, past_context: str) -> str:
        from multimodal_ds.core.llm_client import chat_with_fallback

        prompt = f"""Task: {task_desc}\nData Context:\n{data_context[:1500]}\nPrevious Context:\n{past_context[:500]}\nWorking directory: {self.working_dir}\nWrite Python code. Save all outputs to the current directory."""

        # Use unified LLM client - handles opencode/ and ollama/ prefixes automatically
        try:
            result = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=6000,
                temperature=0.1,
            )
            if result and not result.startswith("[Error:"):
                # Return raw LLM response; extraction will be performed later
                return result.strip()
            logger.warning(f"[CodeAgent] LLM returned error: {result}")
        except Exception as e:
            logger.error(f"[CodeAgent] Code generation failed: {e}")
        logger.warning(f"[CodeAgent] Returning empty code for task — LLM call failed or response unparseable")
        return ""

    def _execute_code(self, code: str, file_paths: Optional[list] = None):
        # Log execution details for debugging
        logger.debug(f"[CodeAgent] Preparing to execute script in {self.working_dir} (script will be written to temporary file)")
        # Log first three lines of the generated script to verify backend setup
        script_preview = "\n".join(code.splitlines()[:3])
        logger.debug(f"[CodeAgent] Script preview (first 3 lines):\n{script_preview}")
        files_before = set(self.working_dir.glob("*"))
        script_path = None
        copied_files = []

        # Copy data files to working dir so code can find them locally
        # Use a dedicated temp subdir inside working_dir (same filesystem → fast rename)
        if file_paths:
            for fp in file_paths:
                src = Path(fp)
                if src.exists():
                    dst = self.working_dir / src.name
                    if not dst.exists():
                        try:
                            # Try hard-link first (instant, zero copy) — works when
                            # src and dst are on the same filesystem
                            try:
                                os.link(src, dst)
                            except (OSError, NotImplementedError):
                                # Cross-filesystem or unsupported — fall back to copy
                                shutil.copy2(src, dst)
                            copied_files.append(dst)
                        except Exception as e:
                            logger.warning(f"[CodeAgent] Failed to copy {src.name}: {e}")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=self.working_dir, delete=False, encoding="utf-8") as f:
                f.write(code)
                script_path = Path(f.name)
            
            run_kwargs = {
                "args": [sys.executable, str(script_path)],
                "cwd": str(self.working_dir),
                "capture_output": True,
                "text": True,
                "timeout": _PROC_TIMEOUT_S
            }
            if sys.platform != "win32":
                run_kwargs["preexec_fn"] = _sandbox_preexec
            
            result = subprocess.run(**run_kwargs)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # Log stderr on error for debugging
            if result.returncode != 0:
                logger.error(f"[CodeAgent] Subprocess exited with code {result.returncode}. Stderr (first 500 chars): {stderr[:500]}")
            # Always include full stderr for debugging, truncate stdout separately
            stderr_section = f"\n[stderr]:\n{stderr}" if stderr else ""
            combined = stdout[:_STDOUT_CHARS] + stderr_section
            
            success = result.returncode == 0 or (result.returncode != 0 and _is_only_warnings(stderr))
            
            if not success:
                logger.error(
                    f"[CodeAgent] Subprocess exited with code {result.returncode}.\n"
                    f"FULL STDERR:\n{stderr}\n"
                    f"STDOUT (first 3000):\n{stdout[:3000]}"
                )
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {_PROC_TIMEOUT_S}s", []
        except Exception as e:
            return False, f"Execution error: {e}", []
        finally:
            if script_path and script_path.exists():
                try: script_path.unlink()
                except Exception: pass
            # Cleanup copied data files to keep sandbox clean
            for cf in copied_files:
                try: cf.unlink()
                except Exception: pass

        files_after = set(self.working_dir.glob("*"))
        new_files = [f.name for f in (files_after - files_before) if f.is_file() and f.suffix != ".py"]
        return success, combined, new_files

    def _execute_with_retry(self, code: str, task_desc: str, data_context: str, file_paths: Optional[list], max_retries: int) -> dict:

        # Fix filename references in generated code before execution
        if file_paths:
            working_files = [str(Path(fp).name) for fp in file_paths]
            code = self._fix_filename_references(code, working_files)

        # Fix common LLM code mistakes before execution
        code = self._fix_common_code_errors(code)

        # Inject safety preamble LAST (after all fixes applied)
        import re as _re
        SAFETY_PREAMBLE = """\
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

"""
        # Strip any existing matplotlib backend setup from generated code to avoid conflicts
        code = _re.sub(r'import matplotlib\s*\n\s*matplotlib\.use\([\'"]Agg[\'"]\)\s*\n', '', code)
        code = _re.sub(r'matplotlib\.use\([\'"]Agg[\'"]\)\s*\n', '', code)
        code = SAFETY_PREAMBLE + code

        # Validate and auto-repair syntax before execution
        code, syntax_ok = self._validate_and_fix_syntax(code)
        if not syntax_ok:
            logger.error("[CodeAgent] Code has unfixable syntax errors — skipping execution")
            return {
                "success": False,
                "code": code,
                "output": "Syntax error in generated code — could not auto-repair.",
                "files_created": [],
                "error": "Syntax error in generated code",
                "retries_used": 0,
            }

        try:
            success, output, files = self._execute_code(code, file_paths)
        except Exception as e:
            logger.error(f"[CodeAgent] Initial execution raised: {e}")
            success, output, files = False, str(e), []

        for attempt in range(max_retries):
            try:
                fix_code = self._generate_fix(code, output, task_desc)
            except Exception as e:
                logger.warning(f"[CodeAgent] Fix generation failed on attempt {attempt + 1}: {e}")
                break

            if fix_code:
                if file_paths:
                    working_files = [str(Path(fp).name) for fp in file_paths]
                    fix_code = self._fix_filename_references(fix_code, working_files)
                fix_code = self._fix_common_code_errors(fix_code)
                fix_code, syntax_ok = self._validate_and_fix_syntax(fix_code)
                if syntax_ok:
                    success, output, files = self._execute_code(fix_code, file_paths)
                    if success:
                        return {
                            "success": True,
                            "code": fix_code,
                            "output": output,
                            "files_created": files,
                            "error": "",
                            "retries_used": attempt + 1,
                        }
                    code = fix_code

        return {
            "success": False,
            "code": code,
            "output": output,
            "files_created": files,
            "error": output,
            "retries_used": max_retries,
        }

    def _generate_fix(self, failed_code: str, error_output: str, task_desc: str) -> str:
        from multimodal_ds.core.llm_client import chat_with_fallback
        import sys

        env_context = f"Python {sys.version.split()[0]}, pandas {self._get_pkg_version('pandas')}, sklearn {self._get_pkg_version('sklearn')}, numpy {self._get_pkg_version('numpy')}"

        prompt = f"""Fix this Python data science script that failed at runtime.

Environment: {env_context}
Task: {task_desc}

Runtime error:
{error_output[:3000]}

Failed script:
```python
{failed_code[-4000:]}
```

Analyze the error carefully and fix the root cause. Common issues to check:
- pandas 2.x: use pd.concat() not df.append(), use .items() not .iteritems()
- pandas 2.x: df.describe(include='all') not include=['object']  
- sklearn: encode categoricals before model.fit()
- matplotlib: call matplotlib.use('Agg') before any other matplotlib import
- train_test_split: always pass X and y as arguments

Return ONE complete fixed Python script in ```python ... ``` fences."""

        try:
            result = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5-coder:7b",
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger. Analyze errors and fix them. Return only the complete fixed script in ```python``` fences."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000,
                temperature=0.0,
            )
            if result and not result.startswith("[Error:"):
                return self._extract_code(result)
        except Exception as e:
            logger.error(f"[CodeAgent] Fix generation failed: {e}")
        return ""

    def _get_pkg_version(self, pkg: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    def _extract_code(self, text: str) -> str:
        import re

        if not text or not text.strip():
            return ""

        # Strip <think>...</think> reasoning blocks (qwen3, deepseek-r1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try ```python ... ``` fence first (allow optional spaces, optional space after backticks, and Windows line endings)
        m = re.search(r'```\s*python\s*\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:
                return code

        # Try any ``` ... ``` fence (any language tag or none)
        m = re.search(r'```.*?\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code and any(
                kw in code
                for kw in ('import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print(')
            ):
                return code

        # Last resort: raw text that looks like Python code. Never return fenced text
        # by filtering out lines that start with markdown code block fences (```).
        cleaned_lines = []
        for line in text.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                continue
            cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines).strip()

        if cleaned_text:
            first_line = cleaned_text.split('\n')[0].strip()
            if any(first_line.startswith(kw) for kw in (
                'import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print('
            )):
                return cleaned_text

        logger.warning("[CodeAgent] Could not extract Python code from LLM response")
        logger.debug(f"[CodeAgent] Raw response (first 300 chars): {text[:300]!r}")
        return ""

    def _get_relevant_memory(self, query: str) -> str:
        memories = self.memory.retrieve(query, n_results=3)
        if not memories:
            return ""
        return "\n".join(m["content"][:200] for m in memories)
