# Repository Audit & Cleanup Report

**Date**: October 21, 2025  
**Status**: Cleanup recommendations ready for execution

---

## 🎯 Audit Summary

This audit identifies deprecated, temporary, and unnecessary files that should be archived or removed to maintain a clean, production-ready repository.

---

## 📋 Files Identified for Cleanup

### 1. **Deprecated Documentation** (Archive)

These are planning/progress docs that served their purpose but are now superseded:

| File | Size | Status | Action | Reason |
|------|------|--------|--------|--------|
| `README_OLD.md` | 625 lines | ❌ Deprecated | Archive | Superseded by current README.md |
| `DOCS_REORGANIZATION.md` | 320 lines | ❌ Deprecated | Archive | Historical planning doc |
| `KG_QUICKSTART_PLAN.md` | 292 lines | ❌ Deprecated | Archive | Planning doc - features now implemented |
| `KG_SOCIAL_SCIENCE_ROADMAP.md` | 638 lines | ❌ Deprecated | Archive | Roadmap doc - Tier 1 now complete |
| `PROGRESS.md` | 238 lines | ❌ Deprecated | Archive | Historical progress tracking |
| `TEST_FIXES.md` | 156 lines | ❌ Deprecated | Archive | Historical test fixes |

**Recommendation**: Move to `archive/docs/`

---

### 2. **Temporary Test Data** (Remove or Archive)

Large data files in root directory that should not be in git:

| File | Size | Type | Action | Reason |
|------|------|------|--------|--------|
| `pol_archive_0.csv` | 186 MB | ⚠️ Data | .gitignore | Too large for git, should be external |
| `pol_archive_4weeks.csv` | 204 MB | ⚠️ Data | .gitignore | Too large for git |
| `pol_archive_60days.csv` | 2.1 MB | ⚠️ Data | .gitignore | Test data |
| `pol_archive_with_dates.csv` | 2.1 MB | ⚠️ Data | .gitignore | Test data |
| `pol_archive_5k.csv` | 303 KB | ✅ OK | Keep | Small sample (useful) |
| `pol_archive_with_users.csv` | 215 KB | ✅ OK | Keep | Small sample (useful) |
| `full_pol.csv` | 5.1 MB | ⚠️ Data | .gitignore | Large test data |
| `test_sentiment_data.csv` | 1 KB | ✅ OK | Keep | Small test file |
| `kg_nodes.csv` | 87 B | 🗑️ Temp | Remove | Temporary output |
| `kg_edges.csv` | 64 B | 🗑️ Temp | Remove | Temporary output |
| `actor_edges.csv` | 89 B | 🗑️ Temp | Remove | Temporary output |
| `actor_metrics.csv` | 46 B | 🗑️ Temp | Remove | Temporary output |

**Recommendations**:
- Remove small temporary outputs (kg_*, actor_*)
- Add large data files to .gitignore
- Keep small samples (<500KB) that are useful for testing

---

### 3. **Temporary Output Directories** (Clean or .gitignore)

Multiple output directories with test results:

| Directory | Purpose | Action | Reason |
|-----------|---------|--------|--------|
| `out/` | Old test outputs | Remove or .gitignore | Legacy test directory |
| `out_actor/` | Actor test outputs | Remove or .gitignore | Temporary test results |
| `out_community/` | Community test outputs | Remove or .gitignore | Temporary test results |
| `out_full/` | Full test outputs | Remove or .gitignore | Temporary test results |
| `out_kg/` | KG test outputs | Remove or .gitignore | Temporary test results |
| `output/` | Current outputs | ✅ Keep but .gitignore | Valid output dir (but don't commit) |

**Recommendation**: Add all `out*` directories to .gitignore, optionally remove local copies

---

### 4. **Notebooks** (Consolidate)

| Directory | Contents | Action | Reason |
|-----------|----------|--------|--------|
| `notebooks/` | 2 older notebooks | Review/Archive | May duplicate examples/ |
| `examples/` | 4 polished notebooks + README | ✅ Keep | Primary examples |

**Files in notebooks/**:
- `contagion_basics.ipynb` - Check if duplicates examples/
- `explore_networks.ipynb` - Check if duplicates examples/

**Recommendation**: Review notebooks/ content, archive if redundant, or move valuable ones to examples/

---

### 5. **CI Log File** (Remove)

| File | Size | Action | Reason |
|------|------|--------|--------|
| `ci_log.txt` | 1662 lines | 🗑️ Remove | Historical CI log (not needed in repo) |

---

### 6. **Empty/New Directories** (Keep but Document)

| Directory | Status | Action |
|-----------|--------|--------|
| `tutorials/` | Empty | Keep | Created for future use |

---

## 🧹 Cleanup Actions

### Priority 1: Archive Deprecated Docs

```bash
# Move planning docs to archive
mv README_OLD.md archive/docs/
mv DOCS_REORGANIZATION.md archive/docs/
mv KG_QUICKSTART_PLAN.md archive/docs/
mv KG_SOCIAL_SCIENCE_ROADMAP.md archive/docs/
mv PROGRESS.md archive/docs/
mv TEST_FIXES.md archive/docs/

# Create archive README
cat > archive/README.md << 'EOF'
# Archive

This directory contains historical documentation that served its purpose but is no longer current:

- **docs/** - Planning documents and roadmaps (now implemented)
- **notebooks/** - Superseded notebooks (examples/ is canonical)
- **test_data/** - Old test datasets
- **output/** - Historical test outputs

These files are preserved for reference but not actively maintained.
EOF
```

### Priority 2: Clean Temporary Files

```bash
# Remove temporary outputs from root
rm -f kg_nodes.csv kg_edges.csv actor_edges.csv actor_metrics.csv

# Remove CI log
rm -f ci_log.txt
```

### Priority 3: Update .gitignore

Add these patterns to `.gitignore`:

```gitignore
# Large data files (>5MB)
pol_archive_0.csv
pol_archive_4weeks.csv
pol_archive_60days.csv
pol_archive_with_dates.csv
full_pol.csv

# Temporary output directories
out/
out_*/
output/

# Temporary root-level outputs
kg_nodes.csv
kg_edges.csv
actor_*.csv
*_edges.csv
*_nodes.csv

# CI logs
ci_log.txt

# Archive (keep in repo but ignore changes)
# archive/ is committed but future changes ignored
```

### Priority 4: Consolidate Notebooks

```bash
# Review notebooks/ directory
ls -la notebooks/

# If redundant with examples/, archive them:
mkdir -p archive/notebooks
mv notebooks/contagion_basics.ipynb archive/notebooks/
mv notebooks/explore_networks.ipynb archive/notebooks/

# Or move valuable ones to examples/ if unique
```

### Priority 5: Document Data Files

Create `DATA.md` to document data management:

```markdown
# Data Management

## Sample Data (In Repo)

Small sample files suitable for git (<500KB):
- `pol_archive_5k.csv` (303 KB) - 5,000 posts sample
- `pol_archive_with_users.csv` (215 KB) - Sample with user IDs
- `test_sentiment_data.csv` (1 KB) - Minimal test data
- `examples/sample_*.csv` - Generated example data

## Large Data Files (External)

Large data files should be stored externally and downloaded:
- `pol_archive_0.csv` (186 MB) - Full archive (not in git)
- `pol_archive_4weeks.csv` (204 MB) - 4-week slice (not in git)
- `pol_archive_60days.csv` (2.1 MB) - 60-day slice (not in git)

**Download instructions**: [Add data download instructions here]

## Output Directories

All output directories are gitignored:
- `output/` - Main output directory
- `out*/` - Legacy test output directories

**Note**: These directories exist locally but are not committed to git.
```

---

## 📊 Impact Assessment

### Before Cleanup

```
Total files: 332
Documentation: 30+ files (6 deprecated)
Data files in root: 12 files (400+ MB)
Output directories: 6 directories
Temporary files: 5+ files
```

### After Cleanup

```
Total files: ~315
Documentation: 24 active files
Data files in root: 3 small samples (<1 MB total)
Output directories: 1 (output/) in .gitignore
Temporary files: 0
Archive: 1 directory (reference only)
```

### Benefits

✅ **Cleaner repository**: Easier to navigate  
✅ **Faster clones**: Remove 400+ MB of data files  
✅ **Clear documentation**: Only active docs visible  
✅ **Better git hygiene**: Output files not tracked  
✅ **Preserved history**: Archive maintains reference  

---

## 🎯 Recommended Execution Order

1. **Create archive/ directory** (done)
2. **Move deprecated docs** to archive/docs/
3. **Remove temporary files** (kg_*, actor_*, ci_log.txt)
4. **Update .gitignore** to prevent future clutter
5. **Review notebooks/** - archive or consolidate
6. **Create DATA.md** - document data management
7. **Commit cleanup**: "chore: archive deprecated docs and clean temporary files"
8. **Update README.md** (if needed) to reflect cleanup

---

## 📝 Documentation Updates Needed

After cleanup, update these files:

### README.md
- ✅ Already clean and up-to-date
- No changes needed

### CONTRIBUTING.md
- Add section on data file management
- Reference DATA.md for large files

### .github/workflows/tests.yml
- Verify tests don't reference archived files
- Ensure tests use examples/ data

---

## ✅ Verification Checklist

After cleanup, verify:

- [ ] All deprecated docs moved to archive/
- [ ] No temporary .csv files in root (except small samples)
- [ ] .gitignore updated with output directories
- [ ] ci_log.txt removed
- [ ] archive/README.md created
- [ ] DATA.md created
- [ ] Tests still pass: `python -m unittest discover tests -v`
- [ ] Examples still work: Check examples/ notebooks
- [ ] Repository size reduced significantly
- [ ] Git status shows clean tree

---

## 🔄 Maintenance Going Forward

### Do's ✅
- Keep output directories in .gitignore
- Store large data files externally
- Use examples/ for canonical notebooks
- Document new features in active docs only
- Archive completed planning docs

### Don'ts ❌
- Don't commit output files to root
- Don't commit large (>5MB) data files
- Don't keep multiple outdated README versions
- Don't accumulate temporary test files
- Don't commit CI logs

---

## 📌 Notes

- **Archive preservation**: Archive directory is committed to git for historical reference
- **Data files**: Large files should be external (S3, GitHub Releases, etc.)
- **Output directories**: Local only, never committed
- **Documentation**: Active docs in root, historical docs in archive/

---

## 🚀 Ready to Execute

All cleanup actions are safe and reversible. Archive preserves all content. Execute Priority 1-3 immediately, then review Priority 4-5.

**Estimated time**: 10 minutes  
**Risk level**: Low (all changes reversible)  
**Git impact**: Cleaner commits, smaller repo size

