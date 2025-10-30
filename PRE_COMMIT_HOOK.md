# Pre-Commit Hook Documentation

## What It Does

The pre-commit hook automatically runs smoke tests before every `git commit`. This prevents deploying broken code to Heroku.

## How It Works

When you run `git commit`, the hook:
1. Runs `python tests/smoke_test.py`
2. If tests pass (exit code 0): Commit proceeds
3. If tests fail (exit code 1): Commit is blocked

## Example Output

### ✅ Tests Pass
```bash
$ git commit -m "Fix bug"
🧪 Running pre-commit smoke tests...

======================================================================
SMOKE TEST - Quick Validation
======================================================================

[1/4] Educational paragraph... ✅ PASS
[2/4] Technical content... ✅ PASS
[3/4] Table structure... ✅ PASS
[4/4] Executive metrics... ✅ PASS

======================================================================
SMOKE TEST RESULTS: 4/4 passed
======================================================================

✅ All smoke tests passed - proceeding with commit

[main abc1234] Fix bug
 1 file changed, 10 insertions(+), 5 deletions(-)
```

### ❌ Tests Fail
```bash
$ git commit -m "Broken change"
🧪 Running pre-commit smoke tests...

======================================================================
SMOKE TEST - Quick Validation
======================================================================

[1/4] Educational paragraph... ❌ FAIL
...

❌ COMMIT BLOCKED: Smoke tests failed

Fix the issues above before committing.
To bypass this check (not recommended): git commit --no-verify
```

## Bypassing the Hook (Emergency Only)

If you absolutely need to commit without running tests:

```bash
git commit --no-verify -m "Emergency fix"
```

**⚠️ WARNING**: Only use `--no-verify` for emergency hotfixes. The hook exists to protect production quality.

## Installation

The hook is already installed at `.git/hooks/pre-commit`. If you need to reinstall:

```bash
chmod +x .git/hooks/pre-commit
```

## Benefits

- 🛡️ **Prevents broken deployments**: Catches syntax errors and major bugs before commit
- ⚡ **Fast feedback**: 30-second smoke test vs 5-minute Heroku build failure
- 💰 **Saves time**: No more "oops" commits followed by immediate fixes
- 📊 **Maintains quality**: Enforces minimum quality standards automatically

## What Tests Run

The smoke test validates:
1. Educational content bullet generation
2. Technical content parsing
3. Table structure handling
4. Executive metrics extraction

See `tests/smoke_test.py` for details.
