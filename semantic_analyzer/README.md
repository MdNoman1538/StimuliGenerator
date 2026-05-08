# Semantic Abstraction Comparator

Analyze the abstraction level differences between NC (Not Concrete) and VC (Very Concrete) stimulus texts using WordNet and spaCy.

## Setup

```bash
cd semantic_analyzer

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download WordNet
python -m nltk.downloader wordnet omw-1.4
```

## Running

```bash
python main.py
```

Server runs on `http://localhost:8001`

## Usage

### Option 1: Direct Access
Open `http://localhost:8001` and paste NC and VC texts to analyze.

### Option 2: From Main App (Programmatic)
From your main generation app, create analysis links:

```python
from urllib.parse import quote

nc_text = "..."  # generated NC stimulus
vc_text = "..."  # generated VC stimulus
label = f"generation_{timestamp}"

analyze_url = f"http://localhost:8001/?ta={quote(nc_text)}&tb={quote(vc_text)}&label={quote(label)}"
```

Then add an "Analyze NC vs VC" button that opens this URL in a new tab:

```html
<button onclick="window.open('{{ analyze_url }}', '_blank')">
  Analyze NC vs VC →
</button>
```

The analyzer will:
1. Auto-populate Text A (NC) and Text B (VC)
2. Extract noun phrases using spaCy
3. Score abstraction levels using WordNet
4. Show side-by-side comparison with delta
5. Save results to `.mat` file

## Output

Results saved to `output_mat/` with timestamp as `{label}_{timestamp}.mat`

Includes:
- All nouns and lemmas
- Noun phrases per slot
- Abstraction level (AL) scores
- WordNet coverage
