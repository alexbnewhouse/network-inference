# Data Management Guide

This document explains how data files are organized and managed in this repository.

---

## 📁 Data Organization

### Sample Data (In Repository)

Small sample files suitable for git (<500KB), used for testing and examples:

| File | Size | Purpose | Location |
|------|------|---------|----------|
| `pol_archive_5k.csv` | ~303 KB | 5,000 posts sample for testing | Root |
| `pol_archive_with_users.csv` | ~215 KB | Sample with user IDs for user-entity networks | Root |
| `test_sentiment_data.csv` | ~1 KB | Minimal test data for sentiment analysis | Root |
| `examples/sample_*.csv` | <100 KB | Generated example data (news, forum, research) | examples/ |

**These files ARE tracked in git** because they're small and useful for tests/examples.

---

### Large Data Files (External)

Large data files should be stored externally and are **NOT committed to git**:

| File | Size | Status | Notes |
|------|------|--------|-------|
| `pol_archive_0.csv` | 186 MB | ❌ Not in git | Full 4chan /pol/ archive |
| `pol_archive_4weeks.csv` | 204 MB | ❌ Not in git | 4-week temporal slice |
| `pol_archive_60days.csv` | 2.1 MB | ❌ Not in git | 60-day slice |
| `pol_archive_with_dates.csv` | 2.1 MB | ❌ Not in git | Sample with timestamps |
| `full_pol.csv` | 5.1 MB | ❌ Not in git | Large test dataset |

**These files are gitignored** to keep repository size manageable.

---

## 🚫 .gitignore Patterns

The following patterns prevent data files from being tracked:

```gitignore
# Most CSV files ignored
*.csv

# But keep small examples
!examples/*.csv

# Small test files explicitly tracked (see above)
# pol_archive_5k.csv (exception)
# pol_archive_with_users.csv (exception)
# test_sentiment_data.csv (exception)

# Output directories always ignored
out/
out_*/
output/
```

---

## 📤 Output Directories

All output directories are gitignored and exist only locally:

| Directory | Purpose | Status |
|-----------|---------|--------|
| `output/` | Main output directory for all analysis | ❌ Not in git |
| `out/` | Legacy test outputs | ❌ Not in git |
| `out_*/` | Various test output directories | ❌ Not in git |

**Note**: These directories will be created automatically when you run commands. They exist locally but are never committed.

---

## 📥 Getting Data

### Option 1: Use Generated Sample Data

Generate small datasets for testing:

```bash
cd examples
python sample_data.py
```

This creates:
- `sample_news.csv` - 100 news headlines
- `sample_forum.csv` - 200 forum posts  
- `sample_research.csv` - 50 research abstracts

### Option 2: Use Included Small Samples

The repository includes small samples you can use immediately:

```bash
# Test with 5K posts sample
python -m src.semantic.kg_cli \
  --input pol_archive_5k.csv \
  --outdir output/test_kg

# Test with user-entity sample
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/test_kg \
  --data pol_archive_with_users.csv \
  --user-col user_id \
  --text-col body
```

### Option 3: Download Large Datasets (External)

For production work with large datasets:

**TODO**: Add instructions for downloading large data files
- Could use GitHub Releases
- Could use external storage (S3, Google Drive, etc.)
- Could provide download script

---

## 🧹 Cleaning Up

### Remove Output Files

```bash
# Remove all output directories
rm -rf out/ out_*/ output/

# They'll be recreated when you run commands
```

### Remove Large Data Files

```bash
# If you want to clean up large files
rm -f pol_archive_0.csv pol_archive_4weeks.csv pol_archive_60days.csv full_pol.csv

# Keep small samples
# Don't delete: pol_archive_5k.csv, pol_archive_with_users.csv, test_sentiment_data.csv
```

---

## 📝 Best Practices

### Do ✅

- **Use small samples** (<500KB) for quick tests
- **Generate example data** with `examples/sample_data.py`
- **Store large files externally** (not in git)
- **Use output/ directory** for all generated files
- **Test on small data first** before scaling up

### Don't ❌

- **Don't commit large files** (>5MB) to git
- **Don't commit output directories** (they're gitignored)
- **Don't commit temporary results** to root directory
- **Don't track individual CSV outputs** in git

---

## 🔍 Checking Git Status

Before committing, verify no large files are staged:

```bash
# Check what's being tracked
git status

# See file sizes
git ls-files | xargs ls -lh | grep -E "[0-9]+M"

# If large files appear, add them to .gitignore
```

---

## 📊 Data File Formats

### CSV Files

All data files should have these columns (minimum):

```python
# For semantic/KG analysis
df = pd.DataFrame({
    'text': ['post 1', 'post 2', ...],  # Required: text content
})

# Optional columns for advanced features
df = pd.DataFrame({
    'text': ['...'],              # Required
    'created_at': [...],          # For temporal analysis
    'user_id': [...],             # For user-entity networks
    'board': [...],               # For group comparisons
    'subject': [...],             # For thread networks
})
```

### Output Files

Networks generate these files:

- `nodes.csv` / `kg_nodes.csv` - Node attributes
- `edges.csv` / `kg_edges.csv` - Edge list with weights
- `graph.graphml` - Graph format for Gephi/Cytoscape
- `*_report.md` - Analysis reports

---

## 🤔 FAQ

**Q: Why are some CSV files tracked and others not?**  
A: Small samples (<500KB) useful for testing are tracked. Large datasets (>5MB) are not.

**Q: Where did my output files go after pulling?**  
A: Output directories are gitignored - they only exist locally. Run commands to regenerate.

**Q: Can I add my own large dataset?**  
A: Yes, but add it to .gitignore so it's not committed. Use external storage for sharing.

**Q: How do I share analysis results?**  
A: Commit the commands you ran (in scripts or docs), not the output files. Others can reproduce.

**Q: What if I accidentally commit a large file?**  
A: Remove it with `git rm --cached filename` and add it to .gitignore immediately.

---

## 📞 Support

Issues with data files?
- Check file size: `ls -lh filename`
- Check git status: `git status`
- Check .gitignore: `cat .gitignore`
- Review this guide: `DATA.md`

For questions, open an issue on GitHub.

---

**Last updated**: October 21, 2025
