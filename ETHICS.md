# Ethics & Privacy Guidelines

**A comprehensive guide for responsible social media research**

This document provides ethical guidelines for researchers using this toolkit to analyze social media data, online communities, and other human-generated text.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [IRB & Institutional Requirements](#irb--institutional-requirements)
3. [Privacy & Anonymization](#privacy--anonymization)
4. [Public vs. Private Data](#public-vs-private-data)
5. [Reporting & Publication](#reporting--publication)
6. [Special Considerations](#special-considerations)
7. [Best Practices Checklist](#best-practices-checklist)
8. [Case Examples](#case-examples)
9. [Resources](#resources)

---

## Core Principles

### 1. Do No Harm

**The primary ethical obligation**: Your research should not cause harm to individuals or communities.

**Potential harms to avoid:**
- Exposing private information
- Enabling harassment or doxxing
- Stigmatizing communities
- Facilitating surveillance
- Misrepresenting findings

### 2. Respect Dignity & Autonomy

Even when analyzing public data:
- People often don't expect their posts to be used in research
- Context collapse: Posts in one community shared with another
- Power imbalances: Researchers have amplification capacity

### 3. Beneficence

**Research should benefit society:**
- Understanding social movements
- Detecting disinformation campaigns
- Studying public health messaging
- Analyzing political discourse

**Balance**: Benefits must outweigh potential risks.

### 4. Justice

**Fair representation and access:**
- Don't cherry-pick quotes to confirm biases
- Report findings accurately, even if unexpected
- Make methods transparent for replication
- Consider who benefits from your research

---

## IRB & Institutional Requirements

### Do You Need IRB Approval?

**It depends on your institution's policy**. Generally:

#### ✅ IRB Often NOT Required:
- Analyzing publicly available data (public tweets, 4chan, public Reddit)
- No interaction with users
- No collection of private information
- Focus on discourse patterns, not individuals

#### ⚠️ IRB Likely Required:
- Private social media (closed Facebook groups, private accounts)
- Direct interaction with users (interviews, surveys)
- Vulnerable populations (minors, patients, prisoners)
- Re-identification risk (small communities, unique quotes)

#### 🔍 Check With Your Institution:
- IRB policies vary by country and university
- US: Some IRBs exempt public social media research
- EU: GDPR may apply even for public data
- **When in doubt, consult your IRB!**

### Documentation for IRB

If seeking approval, prepare:

1. **Data source description**:
   - Platform (Twitter, Reddit, 4chan, etc.)
   - Public vs. private
   - Data collection method
   - Sample size

2. **Privacy protections**:
   - Anonymization procedures
   - Data storage & security
   - Who has access

3. **Risk assessment**:
   - Potential harms
   - Mitigation strategies
   - Benefits to knowledge

4. **Consent considerations**:
   - Terms of Service for platform
   - User expectations of privacy
   - Justification if no consent obtained

---

## Privacy & Anonymization

### Anonymization Techniques

#### 1. Remove Direct Identifiers

**What to remove/hash:**
- Usernames
- Real names
- Email addresses
- IP addresses
- Phone numbers
- Social security numbers
- Any unique identifiers

**Python example:**
```python
import hashlib
import pandas as pd

df = pd.read_csv("data.csv")

# Hash user IDs
df['user_id'] = df['user_id'].apply(
    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
)

# Remove other identifiers
df = df.drop(columns=['email', 'ip_address', 'real_name'])

df.to_csv("data_anonymized.csv", index=False)
```

#### 2. Aggregate Temporal Data

**Problem**: Exact timestamps can be identifying

**Solution**: Round to hour/day/week
```python
# Instead of "2024-01-15 14:23:47"
df['created_at'] = pd.to_datetime(df['created_at']).dt.floor('H')  # Hour
# or
df['created_at'] = pd.to_datetime(df['created_at']).dt.date  # Day only
```

#### 3. K-Anonymity for Small Groups

**Problem**: Rare combinations can identify individuals

**Example**: "38-year-old female professor at MIT studying AI ethics" → easily identifiable

**Solution**: 
- Generalize categories (38 → "35-40")
- Remove low-frequency combinations
- Aggregate rare groups

```python
# Remove users with <5 posts (too little data, high identification risk)
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= 5].index
df = df[df['user_id'].isin(active_users)]
```

### What You Cannot Anonymize

**Be aware:** Some things are hard/impossible to anonymize:

- **Unique writing styles**: Stylometry can identify authors
- **Rare opinions**: Uncommon combinations of views
- **Network structure**: Social graph can reveal identity
- **Cross-platform linkage**: Users may use same username elsewhere

**Mitigation**: 
- Report aggregates, not individuals
- Avoid quoting unique phrases
- Focus on patterns, not exceptions

---

## Public vs. Private Data

### Public Data (Generally Acceptable to Analyze)

#### 4chan/8chan
- ✅ Fully public, intended for anyone to view
- ✅ Most users anonymous (no persistent identity)
- ⚠️ Still aggregate in publications

#### Twitter/X (Public Accounts)
- ✅ Public tweets intended for broad audience
- ✅ Check Terms of Service for academic use
- ⚠️ Users may have privacy expectations
- ⚠️ Protect accounts (<5K followers)

#### Reddit (Public Subreddits)
- ✅ Public posts on open subreddits
- ✅ Generally accepted for research
- ⚠️ Small subreddits may feel private
- ⚠️ Sensitive topics (mental health, trauma)

#### News Comments
- ✅ Intended for public discussion
- ✅ Low privacy expectations

### Private/Restricted Data (Requires Extra Care)

#### Private Facebook Groups
- ❌ Requires consent or IRB approval
- Expectation of limited audience

#### Closed Discord/Slack
- ❌ Requires consent
- Group-specific norms and expectations

#### Direct Messages
- ❌ Never without explicit consent
- High privacy expectations

#### LinkedIn
- ⚠️ Professional context, users may not expect research use
- Check Terms of Service

### Gray Areas

#### Small Online Communities (<100 members)
- May feel private even if technically public
- **Best practice**: Err on side of caution, seek consent

#### Deleted/Private Accounts
- Don't analyze if user has since made account private
- Respect changing privacy preferences

#### Sensitive Topics
- Mental health forums
- Trauma/abuse survivor groups
- LGBTQ+ spaces
- **Even if public, consider not analyzing or seek permission**

---

## Reporting & Publication

### What to Report

#### ✅ DO Report:
- **Aggregated statistics**: "30% of posts mention Entity X"
- **Network patterns**: "Entities A and B are frequently co-mentioned"
- **Temporal trends**: "Discussion increased 50% after Event Y"
- **Community-level findings**: "Board /pol/ shows different sentiment than /int/"

#### ❌ DON'T Report:
- **Individual usernames**: Use "User123" or omit entirely
- **Exact unique quotes**: Searchable → identifiable
- **Small group behavior**: <10 users → identifiable
- **Linking to original posts**: Enables harassment

### Paraphrasing vs. Direct Quotes

**Problem**: Exact quotes are Google-searchable → users identified

**Solutions:**

1. **Paraphrase** when possible:
   - Instead of: "User123 posted: 'Exact quote here'"
   - Use: "Users expressed concerns about X, with one stating [paraphrased content]"

2. **Aggregate quotes**:
   - Instead of: Single user's exact words
   - Use: "Representative quotes include: [combined/modified quotes]"

3. **Alter slightly** (if preserving meaning):
   - Change rare words to common synonyms
   - Generalize specifics ("my Toyota" → "my car")
   - Fix typos/grammar (makes searching harder)

**When to quote directly:**
- Public figures (politicians, celebrities)
- Already widely publicized content
- Historical/archived content
- When exact wording is scientifically critical

### Manuscript Guidance

**Include in your methods section:**
- Data source and dates
- Sampling procedure
- Anonymization steps taken
- IRB approval status (or exemption justification)
- Limitations of privacy protections

**Example methods text:**
> "We analyzed 100,000 posts from the publicly accessible 4chan /pol/ board collected between January-March 2024. User IDs were hashed using SHA-256, and exact quotes were paraphrased to prevent re-identification. This research was determined exempt by [University] IRB under category 4 (secondary analysis of public data)."

---

## Special Considerations

### Vulnerable Populations

#### Minors
- Higher bar for privacy protection
- Even public posts from minors require extra care
- Some journals won't publish minor-sourced data

#### Mental Health Communities
- Suicide forums, self-harm communities
- High sensitivity even if public
- Consider: Is research benefit worth potential harm?

#### Extremist/Terrorist Content
- Security considerations for researchers
- Ethical questions about amplifying dangerous ideologies
- May require special data handling protocols

### Platform-Specific Issues

#### 4chan/8chan
- **Highly anonymous**: Low re-identification risk
- **Offensive content**: Content warnings in publications
- **Ephemeral**: Posts deleted after ~24 hours (archive.is captures)
- **No accounts**: True anonymity (best case for privacy)

#### Reddit
- **Persistent identities**: Users build reputation over time
- **Small communities**: Easy to identify regular posters
- **Deleted posts**: Respect deletion (don't use archived versions)

#### Twitter/X
- **Public figures**: Different standards (less privacy protection)
- **Doxxing risk**: Right-wing researchers targeted
- **API access**: Check Terms of Service for research use

### International Considerations

#### GDPR (European Union)
- Applies to EU residents' data
- "Right to be forgotten" must be respected
- May require Data Protection Impact Assessment (DPIA)
- Lawful basis: "Legitimate interest" or "Public interest research"

#### Other Jurisdictions
- China: Data localization requirements
- Brazil: LGPD (similar to GDPR)
- California: CCPA (consumer privacy act)

**Recommendation**: If analyzing international data, consult legal expert.

---

## Best Practices Checklist

Before starting research:

- [ ] **Determine if IRB review needed** (consult your institution)
- [ ] **Check platform Terms of Service** for research use
- [ ] **Assess re-identification risk** (public figures vs. private individuals)
- [ ] **Plan anonymization strategy** (hashing, aggregation, generalization)
- [ ] **Consider potential harms** (to individuals, communities, society)
- [ ] **Document data handling** (for IRB, publications, collaborators)

During analysis:

- [ ] **Use anonymized IDs** (never expose original usernames)
- [ ] **Aggregate small groups** (combine groups <10 users)
- [ ] **Test re-identification risk** (can you identify individuals? If yes, add protections)
- [ ] **Store data securely** (encrypted, access-controlled)

When reporting:

- [ ] **Paraphrase quotes** (or alter to prevent searching)
- [ ] **Report aggregates** (not individual behaviors)
- [ ] **Avoid linking to posts** (no URLs to original content)
- [ ] **Use general descriptions** ("Users on extremist forums" not "Users on [specific site]")
- [ ] **Include ethics statement** in methods section
- [ ] **Respect platform communities** (don't stigmatize unnecessarily)

---

## Case Examples

### Example 1: 4chan /pol/ Discourse Analysis ✅

**Scenario**: Analyzing narratives about geopolitical events on /pol/

**Ethical approach:**
- ✅ Public, anonymous board → Low privacy risk
- ✅ No persistent user IDs → Can't track individuals
- ✅ Focus on discourse patterns, not users
- ✅ Paraphrase offensive content in quotes
- ✅ IRB exemption (public data, no interaction)

**Reporting**: "Analysis of 1M posts from public political forums revealed..."

### Example 2: Reddit Mental Health Subreddit ⚠️

**Scenario**: Studying support networks in depression subreddit

**Ethical concerns:**
- ⚠️ Vulnerable population (depression)
- ⚠️ May not expect research use (feels private)
- ⚠️ Re-identification risk (personal stories)

**Ethical approach:**
- ✅ Seek IRB approval (vulnerable population)
- ✅ High anonymization (remove all usernames, hash IDs)
- ✅ No direct quotes (highly identifying)
- ✅ Aggregate-only reporting
- ✅ Consider benefit: Does research help community?
- ⚠️ Alternative: Seek consent from mods/users

**Reporting**: "Analysis of anonymous posts in mental health communities..."

### Example 3: Twitter Political Campaign ✅

**Scenario**: Tracking sentiment during election

**Ethical approach:**
- ✅ Public tweets → Generally acceptable
- ✅ Political speech → Lower privacy expectations
- ⚠️ Protect small accounts (<5K followers)
- ⚠️ Don't amplify harassment
- ✅ IRB may not be required (check institution)

**Reporting**: Differentiate public figures (OK to name) from regular users (anonymize)

---

## Resources

### IRB & Ethics Guidance

- **AoIR Ethics Guidelines**: [https://aoir.org/ethics/](https://aoir.org/ethics/)
- **SAGE Research Ethics**: Social media research standards
- **NIH Human Subjects Research**: [https://grants.nih.gov/policy/humansubjects](https://grants.nih.gov/policy/humansubjects)
- **EU GDPR**: [https://gdpr.eu/](https://gdpr.eu/)

### Academic Literature

- Zimmer, M. (2010). "But the data is already public": On the ethics of research in Facebook
- boyd, d. & Crawford, K. (2012). Critical questions for big data
- Vitak, J., et al. (2017). Balancing audience and privacy on social media
- Fiesler, C. & Proferes, N. (2018). Participant perceptions of Twitter research ethics

### Platform-Specific Guidance

- **Twitter Developer Agreement**: Academic research use
- **Reddit Data API**: Rules for research
- **Facebook/Meta Research**: Requires partnership for private data

### Privacy Tools

- **Amnesia**: K-anonymity tool
- **ARX**: Data anonymization software
- **Differential Privacy**: Google, Apple frameworks

---

## Summary

### Key Takeaways

1. **Public ≠ Unethical to analyze**, but:
   - Users may not expect research use
   - Aggreg ate in publications
   - Protect vulnerable populations

2. **Always anonymize**:
   - Hash user IDs
   - Remove direct identifiers
   - Aggregate small groups

3. **Report responsibly**:
   - Paraphrase quotes
   - Focus on patterns, not individuals
   - Include ethics statement

4. **When in doubt**:
   - Consult your IRB
   - Err on side of caution
   - Ask: "Would I want my posts used this way?"

### Golden Rule

**"Treat your participants the way you would want to be treated if your online activity were being studied."**

---

## Questions?

For institutional-specific guidance:
- **IRB office**: Contact your university's IRB
- **Legal counsel**: For GDPR, Terms of Service questions
- **Ethics committee**: Professional associations (ASA, APA, etc.)

For toolkit-specific privacy features:
- See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for anonymization examples
- See [GETTING_STARTED.md](GETTING_STARTED.md) for secure data handling

**Remember**: Ethics is not just about following rules—it's about doing research responsibly and respectfully.
