from __future__ import annotations

import base64
import json
import logging
import re
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid configuring logging multiple times in Streamlit reruns
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Configuration & Constants ---
st.set_page_config(page_title="MIMIC Patient Data Viewer", layout="wide", initial_sidebar_state="expanded")


# --- Data Loading and Caching ---
@st.cache_data
def load_ndjson_data(file: str) -> pd.DataFrame:
    """
    Load a newline-delimited JSON file into a DataFrame.

    Args:
        file: Path to the NDJSON file on disk.

    Returns:
        A pandas DataFrame of normalized JSON records.
    """
    with open(file, "r") as f:
        lines = [line for line in f.read().splitlines() if line]
    records = [json.loads(line) for line in lines]
    return pd.json_normalize(records)

@st.cache_data
def load_patient_data(uploaded_file) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load and parse an uploaded patient JSON file into DataFrames by resource type.

    Args:
        uploaded_file: Streamlit uploaded file-like object containing patient JSON.

    Returns:
        Mapping of resource type to DataFrame, with a special key 'patient_id'.
        Returns None if parsing fails or file is not provided.
    """
    if uploaded_file is None:
        return None

    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(file_content)

        patient_id = data.get('patient_id', 'Unknown Patient')
        fhir_data = data.get('data', {}) or {}

        processed_data: Dict[str, pd.DataFrame] = {'patient_id': patient_id}  # type: ignore[assignment]

        for resource_type, records in fhir_data.items():
            if records:
                processed_data[resource_type] = pd.json_normalize(records, max_level=3)
            else:
                processed_data[resource_type] = pd.DataFrame()

        return processed_data
    except Exception as e:
        logger.exception("Error loading or parsing uploaded patient file")
        st.error(f"Error loading or parsing file: {e}")
        return None


def load_patient_data_file(path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load a patient JSON from a local file path into the same structure as load_patient_data.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)

        patient_id = data.get('patient_id', 'Unknown Patient')
        fhir_data = data.get('data', {}) or {}

        processed_data: Dict[str, pd.DataFrame] = {'patient_id': patient_id}  # type: ignore[assignment]
        for resource_type, records in fhir_data.items():
            processed_data[resource_type] = pd.json_normalize(records, max_level=3) if records else pd.DataFrame()
        return processed_data
    except Exception as e:
        logger.exception("Error loading or parsing patient file path: %s", path)
        st.error(f"Error loading or parsing file: {e}")
        return None


def list_default_patients(dir_path: str) -> List[Tuple[str, str]]:
    """
    List available default patient files.

    Returns list of tuples (path, patient_id_from_filename).
    """
    try:
        import os
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]
        results: List[Tuple[str, str]] = []
        for p in sorted(files):
            # Expect filenames like patient_<id>.json
            fname = p.split('/')[-1]
            patient_id = fname.replace('patient_', '').replace('.json', '')
            results.append((p, patient_id))
        return results
    except Exception as e:
        logger.warning("Could not list default patients in %s: %s", dir_path, e)
        return []


def pseudonymize_patient_id(patient_id: str) -> str:
    """
    Deterministically map a patient_id to a fake but human-friendly name.
    """
    first_names = [
        'Alex', 'Riley', 'Jordan', 'Taylor', 'Casey', 'Avery', 'Quinn', 'Morgan', 'Drew', 'Reese',
        'Rowan', 'Jamie', 'Skyler', 'Peyton', 'Cameron', 'Logan', 'Emerson', 'Hayden', 'Sawyer', 'Elliot'
    ]
    last_names = [
        'Carter', 'Morgan', 'Brooks', 'Reed', 'Parker', 'Hayes', 'Bennett', 'Sullivan', 'Campbell', 'Collins',
        'Gray', 'Morris', 'Mitchell', 'Bailey', 'Jensen', 'Wells', 'Rowe', 'Dawson', 'Hudson', 'Jasper'
    ]
    h = hashlib.md5(patient_id.encode('utf-8')).hexdigest()
    fi = int(h[:8], 16) % len(first_names)
    li = int(h[8:16], 16) % len(last_names)
    return f"{first_names[fi]} {last_names[li]}"

# --- Helper Functions ---
def safe_get(dct: Any, keys: Sequence[Any], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary or list.

    Args:
        dct: The nested structure (dict/list) to traverse.
        keys: Sequence of keys or indices to follow.
        default: Fallback value if any key/index is missing.

    Returns:
        The nested value if present; otherwise, the provided default.
    """
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, TypeError, IndexError):
            return default
    return dct

def format_value(val: Any) -> str:
    """Format numeric-like values cleanly; return string for non-numeric."""
    try:
        num = float(val)
        return str(int(num)) if num.is_integer() else f"{num:.1f}"
    except (ValueError, TypeError):
        return str(val)

def get_display_name(row: Mapping[str, Any], key_list: Sequence[Any]) -> str:
    """Safely extract a display name from typical FHIR structures with fallback."""
    display = safe_get(row, key_list)
    return str(display) if display else str(row.get(key_list[0], "N/A"))

def parse_date(value: str) -> Optional[datetime]:
    """
    Try parsing a string into a datetime object if it looks like a date.

    Uses a curated set of formats; if none match, falls back to pandas parsing.
    """
    date_formats = [
        "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%d-%b-%Y", "%Y-%m-%d",
        "%m-%d-%Y", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%Y/%m/%d",
    ]
    datetime_formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%b %d, %Y %H:%M:%S", "%B %d, %Y %H:%M:%S", "%b %d, %Y %H:%M", "%B %d, %Y %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
        "%b %d %Y %H:%M:%S", "%B %d %Y %H:%M:%S", "%b %d %Y %H:%M", "%B %d %Y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%b %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M:%S %p", "%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p", "%b %d %Y %I:%M:%S %p", "%B %d %Y %I:%M:%S %p",
        "%b %d %Y %I:%M %p", "%B %d %Y %I:%M %p",
        "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    ]
    for fmt in [*datetime_formats, *date_formats]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Fallback: try pandas flexible parser
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.to_pydatetime()
    except Exception:
        return None

def format_datetime(value: Any, format_str: str = "%m-%d-%Y %H:%M:%S") -> str:
    """Convert a value to formatted datetime string if possible, else return str(value)."""
    try:
        dt = parse_date(str(value))
        return dt.strftime(format_str) if dt else str(value)
    except Exception:
        return str(value)

def get_latest_vital(df: pd.DataFrame, vital_name: str) -> Any:
    """Get the most recent value for a specific vital sign from a DataFrame."""
    latest = (
        df[df["Vital"] == vital_name]
        .sort_values(by="Timestamp", ascending=False)
        .head(1)
    )
    return latest.iloc[0]["Value"] if not latest.empty else "N/A"

# --- Condition grouping
def get_condition_group(condition: str, icd_code: Optional[str] = None) -> str:
    """
    Classify ICD-9 condition as acute, chronic, or unspecified based on description and code.
    
    Args:
        condition: ICD-9 condition description
        icd_code: Optional ICD-9 code for additional context
    
    Returns:
        str: 'acute', 'chronic', or 'unspecified'
    """
    if not condition or not isinstance(condition, str):
        return "unspecified"
    
    condition_clean = re.sub(r'[^\w\s]', ' ', condition.lower().strip())
    
    # Expanded keyword sets with more comprehensive terms
    acute_keywords = [
        "acute", "sudden", "rapid", "abrupt", "initial", "new onset",
        "emergency", "urgent", "severe", "crisis", "attack", "episode",
        "flare", "exacerbation", "first", "primary"
    ]
    
    chronic_keywords = [
        "chronic", "persistent", "long-term", "longstanding", "recurrent",
        "ongoing", "continuous", "permanent", "established", "old",
        "history of", "sequela", "late effect", "residual", "stable"
    ]
    
    unspecified_keywords = [
        "unspecified", "nonspecific", "not specified", "not otherwise specified",
        "nos", "unknown", "undetermined", "other", "unqualified"
    ]
    
    # Check for explicit temporal indicators first (highest priority)
    if any(keyword in condition_clean for keyword in acute_keywords):
        return "acute"
    elif any(keyword in condition_clean for keyword in chronic_keywords):
        return "chronic"
    elif any(keyword in condition_clean for keyword in unspecified_keywords):
        return "unspecified"
    
    # Use ICD-9 code patterns if available
    if icd_code:
        # Many unspecified conditions end in .9
        if icd_code.endswith('.9') or icd_code.endswith('9'):
            return "unspecified"
    
    # Clinical condition patterns (disease-specific logic)
    condition_patterns = get_condition_specific_patterns()
    
    for pattern_type, patterns in condition_patterns.items():
        if any(pattern in condition_clean for pattern in patterns):
            return pattern_type
    
    # Default to unspecified if no clear indicators
    return "unspecified"


def get_condition_specific_patterns() -> Dict[str, List[str]]:
    """
    Disease-specific patterns that indicate temporal nature.
    Based on clinical knowledge of conditions that are typically acute or chronic.
    """
    return {
        "acute": [
            # Infections (typically acute unless specified otherwise)
            "pneumonia", "bronchitis", "gastroenteritis", "appendicitis",
            "cellulitis", "abscess", "sepsis", "meningitis",
            # Injuries and trauma
            "fracture", "laceration", "contusion", "sprain", "strain",
            "burn", "poisoning", "overdose",
            # Acute medical events
            "myocardial infarction", "stroke", "embolism", "thrombosis",
            "hemorrhage", "infarction", "ischemia"
        ],
        "chronic": [
            # Chronic diseases
            "diabetes mellitus", "hypertension", "copd", "emphysema",
            "cirrhosis", "arthritis", "osteoporosis", "dementia",
            "parkinson", "multiple sclerosis", "epilepsy", "migraine",
            "asthma", "bronchiectasis", "fibrosis", "nephritis",
            # Cancer (generally chronic management)
            "carcinoma", "sarcoma", "lymphoma", "leukemia", "neoplasm malignant"
        ]
    }


def get_condition_group_with_confidence(condition: str, icd_code: Optional[str] = None) -> tuple[str, float]:
    """
    Enhanced version that returns classification with confidence score.
    
    Returns:
        tuple: (classification, confidence_score)
        confidence_score: 0.0-1.0, where 1.0 is highest confidence
    """
    classification = get_condition_group(condition, icd_code)
    
    condition_clean = re.sub(r'[^\w\s]', ' ', condition.lower().strip())
    
    # High confidence keywords
    high_confidence_terms = {
        "acute": ["acute", "sudden", "rapid", "emergency"],
        "chronic": ["chronic", "persistent", "long-term", "longstanding"],
        "unspecified": ["unspecified", "not specified", "nos"]
    }
    
    # Check for high confidence indicators
    for category, terms in high_confidence_terms.items():
        if any(term in condition_clean for term in terms):
            if category == classification:
                return classification, 0.9
    
    # Medium confidence for disease-specific patterns
    condition_patterns = get_condition_specific_patterns()
    for pattern_type, patterns in condition_patterns.items():
        if any(pattern in condition_clean for pattern in patterns):
            if pattern_type == classification:
                return classification, 0.7
    
    # Low confidence for defaults
    return classification, 0.3

# --- Vital and Lab grouping (optimized)

# Global, immutable categories for Lab sorting (tuples for faster iteration)
LAB_SORTING_CATEGORIES: Dict[str, Dict[str, Any]] = {
    'Chemistry_Basic': {
        'keywords': (
            'glucose', 'glu-', 'sodium', 'na+', 'potassium',
            'chloride', 'cl-', 'bicarbonate', 'hco3', 'tco2', 'co2',
            'bun', 'urea', 'creatinine', 'egfr', 'gfr', 'anion gap', 'osmolality'
        ),
        'priority': 1
    },
    'Chemistry_Extended': {
        'keywords': (
            'albumin', 'prealbumin', 'total protein', 'protein', 'calcium', 'ionized calcium',
            'phosph', 'phosphate', 'phosphorus', 'magnesium', 'iron', 'ferritin', 'transferrin', 'tibc', 'uibc',
            'b12', 'cobalamin', 'folate', 'folic acid', 'vitamin d', '25-oh d', '25 hydroxy', '1,25 dihydroxy',
            'tsh', 'thyroid stimulating hormone', 'free t4', 'free thyroxine', 'free t3', 'triiodothyronine', 'vitamin'
        ),
        'priority': 2
    },
    'Liver_Related': {
        'keywords': (
            'alt', 'alanine', 'ast', 'aspartate', 'alkaline phosphatase', 'alkaline', 'phosphatase',
            'bilirubin', 'bili', 'direct bilirubin', 'indirect bilirubin', 'conjugated', 'unconjugated',
            'ggt', 'gamma glutamyl', 'ammonia', 'hepatic', 'liver', 'alp'
        ),
        'priority': 3
    },
    'Cardiac_Related': {
        'keywords': (
            'troponin', 'trop i', 'trop t', 'hs troponin', 'high sensitivity troponin',
            'ck', 'creatine kinase', 'ck-mb', 'myoglobin',
            'bnp', 'pro-bnp', 'nt-probnp', 'natriuretic', 'cardiac', 'heart',
            'ldh', 'lactate dehydrogenase'
        ),
        'priority': 4
    },
    'Hematology_Complete': {
        'keywords': (
            'wbc', 'white blood', 'leukocyte', 'rbc', 'red blood', 'hemoglobin', 'hgb', 'hematocrit', 'hct',
            'platelet', 'plt', 'mpv', 'nrbc', 'mcv', 'mch', 'mchc', 'rdw', 'reticulocyte', 'retic', 'ipf'
        ),
        'priority': 5
    },
    'Hematology_Differential': {
        'keywords': (
            'neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil', 'bands', 'segs',
            'metamyelocyte', 'myelocyte', 'immature granulocyte',
            'absolute', 'differential', 'basos', 'eos', 'lymphs', 'monos', 'neuts', 'atypical lymphocyte'
        ),
        'priority': 6
    },
    'Microbiology': {
        'keywords': (
            # Culture-related
            'culture', 'anaerobic', 'aerobic', 'bottle', 'blood culture', 'urine culture', 'wound culture',
            'fluid culture', 'tissue', 'respiratory culture', 'fecal culture', 'stool', 'sputum', 'gram stain',
            'bcx', 'ucx', 'blood cx', 'urine cx', 'resp cx',
            # Specific screens and PCR
            'mrsa', 'vre', 'c. difficile', 'difficile', 'pcr', 'naat', 'covid', 'sars-cov-2',
            # Stains and parasites
            'ova', 'parasites', 'acid fast', 'afsmear', 'cyclospora', 'microsporidia',
            'cryptosporidium', 'giardia',
            # Fungal and viral
            'fungal', 'virus', 'viral', 'ebv', 'cmv', 'hcv', 'hiv', 'varicella', 'rubeola',
            'rubella', 'toxoplasma', 'legionella',
            # Serology-specific (to distinguish from general immune)
            'igg', 'igm', 'iga', 'ige', 'ebna', 'vca', 'serology',
            # Organisms (common in micro reports)
            'enterobacter', 'escherichia', 'staphylococcus', 'enterococcus', 'acinetobacter',
            'coagulase negative',
            # Common antibiotics for susceptibility panels (specific to micro context)
            'levofloxacin', 'clindamycin', 'erythromycin', 'meropenem', 'cefepime',
            'piperacillin', 'ciprofloxacin', 'ceftriaxone', 'ceftazidime', 'tobramycin',
            'gentamicin', 'trimethoprim', 'vancomycin', 'rifampin', 'oxacillin',
            'tetracycline', 'daptomycin', 'ampicillin', 'cefazolin', 'amikacin',
            'sulfa', 'susceptibility', 'mic '
        ),
        'priority': 7
    },
    'Blood_Gas': {
        'keywords': (
            'abg', 'vbg', 'pco2', 'pco₂', 'po2', 'po₂', 'hco3', 'bicarbonate',
            'co2 pressure', 'o2 pressure', 'o2 saturation', 'oxygen', 'base excess', 'a-a gradient',
            'arterial', 'venous', 'lactate', 'lactic', 'carboxyhemoglobin', 'methemoglobin', 'sao2', 'svo2'
        ),
        'priority': 8
    },
    'Coagulation': {
        'keywords': (
            'ptt', 'inr', 'prothrombin', 'partial thromboplastin',
            'coagulation', 'clotting', 'anti-xa', 'd-dimer', 'fibrinogen', 'thrombin time', 'act '
        ),
        'priority': 9
    },
    'Hormones_Endocrine': {
        'keywords': (
            'hormone', 'parathyroid', 'pth', 'thyroid', 'tsh',
            'cortisol', 'acth', 'insulin', 'hba1c', 'hemoglobin a1c',
            'prolactin', 'testosterone', 'estradiol', 'progesterone', 'fsh',
            'renin', 'aldosterone', 'growth hormone', 'igf-1'
        ),
        'priority': 10
    },
    'Inflammatory_Immune': {
        'keywords': (
            'crp', 'hs-crp', 'procalcitonin', 'esr', 'sed rate', 'sedimentation rate', 'complement',
            'immunoglobulin', 'igg ', 'iga ', 'igm ', 'ige ', 'rheumatoid', 'anti-ccp',
            'ana', 'antinuclear'
        ),
        'priority': 11
    },
    'Enzymes_Other': {
        'keywords': (
            'lipase', 'amylase', 'aldolase', 'enzyme', 'kinase', 'transferase',
            'dehydrogenase', 'haptoglobin', 'g6pd', 'cholinesterase'
        ),
        'priority': 12
    },
    'Urine_Analysis': {
        'keywords': (
            'urine', 'urinalysis', 'specific gravity', 'ketone', 'nitrite',
            'leukocyte', 'epithelial', 'bacteria', 'yeast', 'cast', 'urobilinogen',
            'proteinuria', 'microalbumin', 'albumin/creatinine ratio', 'acr', 'urine sodium', 'urine creatinine',
            'urine osmolality', 'glucose urine', 'bilirubin urine', 'rbc/hpf', 'wbc/hpf'
        ),
        'priority': 13
    },
    'Drugs_Toxicology': {
        'keywords': (
            'vancomycin', 'digoxin', 'lithium', 'drug', 'toxic', 'level',
            'therapeutic', 'peak', 'trough',
            'acetaminophen', 'apap', 'salicylate', 'ethanol', 'barbiturate', 'benzodiazepine',
            'opiates', 'opiate', 'amphetamines', 'cocaine', 'phenytoin', 'valproic', 'carbamazepine',
            'theophylline', 'phenobarbital'
        ),
        'priority': 14
    },
    'Specimen_Info': {
        'keywords': (
            'hold', 'tube', 'collection', 'specimen', 'temperature', 'appearance',
            'color', 'mucous', 'edta', 'green top', 'hemolyzed', 'lipemic', 'icteric', 'recollect', 'specimen source'
        ),
        'priority': 15
    },
}

@st.cache_data
def _clean_lab_name_cached(lab_name: str) -> str:
    cleaned = lab_name.lower().strip()
    cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\s*#/\w+', '', cleaned)
    return cleaned.strip()

@st.cache_data
def _find_lab_category_cached(cleaned_name: str) -> str:
    """Find best lab category using simple keyword counts; fall back to 'Uncategorized'."""
    best_cat = 'Uncategorized'
    best_count = 0
    for cat, info in LAB_SORTING_CATEGORIES.items():
        kws = info['keywords']
        count = sum(1 for kw in kws if kw in cleaned_name)
        if count > best_count or (count == best_count and best_cat == 'Uncategorized'):
            if count > 0:
                best_cat = cat
                best_count = count
    return best_cat

# --- Lab Sorting

def clean_lab_name(lab_name: str) -> str:
    return _clean_lab_name_cached(lab_name)

def find_lab_category(lab_name: str) -> str:
    cleaned_name = _clean_lab_name_cached(lab_name)
    return _find_lab_category_cached(cleaned_name)

def sort_labs_by_category(lab_list: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for lab in lab_list:
        cat = find_lab_category(lab)
        grouped.setdefault(cat, []).append(lab)
    # Priorities with 'Uncategorized' last
    priorities = {cat: info['priority'] for cat, info in LAB_SORTING_CATEGORIES.items()}
    priorities['Uncategorized'] = 999
    ordered: Dict[str, List[str]] = {}
    for cat in sorted(grouped.keys(), key=lambda c: priorities.get(c, 999)):
        ordered[cat] = sorted(grouped[cat], key=lambda s: s.lower())
    return ordered


VITAL_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Blood Pressure": {
        "keywords": (
            "systolic", "diastolic", "mean arterial", "map", "blood pressure",
            "arterial line", "art line", "a-line", "invasive blood pressure", "non-invasive blood pressure", "nibp", "ibp"
        ),
        "priority": 1,
    },
    "Heart/Respiratory Rate": {
        "keywords": (
            # Heart
            "heart rate", "pulse", "pulse rate", "ventricular rate", "atrial rate", "bpm",
            # Respiratory
            "respiratory rate", "breaths", "breaths/min", "vent rate", "resp rate", "spontaneous rate"
        ),
        "priority": 2,
    },
    "Temperature": {
        "keywords": (
            "temperature", "temp", "tympanic", "oral", "rectal", "axillary", "core", "esophageal",
            "bladder", "celsius", "fahrenheit", "core temp"
        ),
        "priority": 4,
    },
    "Oxygenation": {
        "keywords": ("spo2", "sp02", "oxygen saturation", "o2 sat", "pulse ox", "sao2", "oximetry"),
        "priority": 5,
    },
    "Capnography": {
        "keywords": ("etco2", "end tidal", "end tidal co2", "capnograph", "capnogram", "petco2"),
        "priority": 6,
    },
    "Oxygen Therapy/Delivery": {
        "keywords": (
            "fio2", "o2 flow", "oxygen flow", "l/min", "nasal cannula", "nonrebreather", "nrb",
            "venturi", "trach collar", "high flow", "hf nc", "hfnc", "vapotherm", "face mask", "trach"
        ),
        "priority": 7,
    },
    "Ventilation/Device Settings": {
        "keywords": (
            "ventilator", "mode", "peep", "pip", "pressure support", "tidal volume", "tvt",
            "minute ventilation", "insp time", "i:e ratio", "rate (vent)",
            "simv", "pcv", "prvc", "psv", "aprv", "cpap", "bipap"
        ),
        "priority": 8,
    },
    "Hemodynamics (Advanced)": {
        "keywords": ("cvp", "cardiac output", "cardiac index", "svr", "pvr", "svv", "ppv", "pap", "pas", "pad", "pcwp", "pawp", "stroke volume"),
        "priority": 9,
    },
    "ECG/Rhythm & Intervals": {
        "keywords": (
            "rhythm", "telemetry", "qtc", "qtcf", "qt interval", "pr interval", "p-r", "rr interval", "qrs", "st segment", "st elevation", "st depression",
            "ectopy", "afib", "atrial fibrillation", "aflutter", "pvc", "pac"
        ),
        "priority": 10,
    },
    "Neurologic": {
        "keywords": ("gcs", "glasgow", "gcs eye", "gcs motor", "gcs verbal", "pupil", "pupil size", "pupil reactivity", "pupill", "npi", "rass", "richmond", "cam-icu", "sedation", "avpu"),
        "priority": 11,
    },
    "Pain": {
        "keywords": ("pain", "pain score", "nrs", "vas", "cpot", "faces"),
        "priority": 12,
    },
    "Anthropometrics": {
        "keywords": ("height", "weight", "admission weight", "dry weight", "bmi", "body mass", "head circumference", "mid-arm circumference"),
        "priority": 13,
    },
    "Point-of-Care Glucose": {
        "keywords": ("poc glucose", "fingerstick", "accu", "accucheck", "accu-chek", "fs glucose", "bedside glucose", "glucometer"),
        "priority": 14,
    },
    "Fluid Balance (I&O)": {
        "keywords": ("intake", "output", "i&o", "ins", "outs", "net", "balance", "urine output", "uop", "urinary", "ostomy", "drain", "chest tube", "ng output", "emesis", "stool", "po intake", "iv intake"),
        "priority": 15,
    },
}

def vital_category_priority(category: str) -> int:
    return VITAL_CATEGORIES.get(category, {}).get('priority', 999)

@st.cache_data
def classify_vital(name: str) -> str:
    n = name.lower().strip()
    # Evaluate categories in priority order
    for category in sorted(VITAL_CATEGORIES.keys(), key=vital_category_priority):
        if any(k in n for k in VITAL_CATEGORIES[category]['keywords']):
            return category
    return 'Other'

def order_names_within_category(category: str, names: List[str]) -> List[str]:
    if category == 'Blood Pressure':
        def bp_key(x: str) -> tuple:
            xl = x.lower()
            if 'systolic' in xl:
                p = 0
            elif 'diastolic' in xl:
                p = 1
            elif ('map' in xl) or ('mean arterial' in xl):
                p = 2
            else:
                p = 3
            return (p, xl)
        return sorted(names, key=bp_key)
    return sorted(names, key=lambda s: s.lower())


# --- Data Stitching Logic ---
# @st.cache_data
def prepare_grouped_conditions(enc_conditions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a grouped conditions DataFrame with columns Condition, Code, Group.

    Mirrors the UI’s logic to ensure identical results.
    """
    if enc_conditions_df is None or enc_conditions_df.empty:
        return pd.DataFrame(columns=["Condition", "Code", "Group"])  # empty structure

    cond_list = [
        safe_get(row["code.coding"], [0, "display"], "N/A")
        for _, row in enc_conditions_df.iterrows()
    ]
    code_list = [
        safe_get(row["code.coding"], [0, "code"], "N/A")
        for _, row in enc_conditions_df.iterrows()
    ]
    cond_df = pd.DataFrame({"Condition": cond_list, "Code": code_list})
    cond_df["Group"] = cond_df["Condition"].apply(get_condition_group)
    return cond_df


def prepare_lab_sorted_groups(labs_clean_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize labs into groups using keyword matching and return mapping of
    category -> list of lab names in display order.
    """
    if labs_clean_df is None or labs_clean_df.empty:
        return {}

    unique_labs = sorted(labs_clean_df["Lab Test"].unique().tolist())
    return sort_labs_by_category(unique_labs)


def stitch_encounter_data(_data, locations_map, med_map):
    """Stitches all related patient data into a single encounters DataFrame."""
    
    # Combine all encounter types
    enc_df = _data.get('MimicEncounter', pd.DataFrame())
    enc_df['_type'] = 'Inpatient'
    enc_ed_df = _data.get('MimicEncounterED', pd.DataFrame())
    enc_ed_df['_type'] = 'Emergency'
    enc_icu_df = _data.get('MimicEncounterICU', pd.DataFrame())
    enc_icu_df['_type'] = 'ICU'
    
    all_enc_df = pd.concat([enc_df, enc_ed_df, enc_icu_df], ignore_index=True)
    if all_enc_df.empty:
        return pd.DataFrame()
        
    all_enc_df.sort_values(by='period.start', inplace=True, ascending=False)
    all_enc_df.reset_index(drop=True, inplace=True)

    # Prepare for stitching
    stitched_data = []

    # Get all data sources, handling missing ones with empty dataframes
    all_cond_df = pd.concat([_data.get("MimicCondition", pd.DataFrame()), _data.get("MimicConditionED", pd.DataFrame())])
    all_proc_df = pd.concat([_data.get('MimicProcedure', pd.DataFrame()), _data.get('MimicProcedureED', pd.DataFrame()), _data.get('MimicProcedureICU', pd.DataFrame())])
    med_req_df = _data.get('MimicMedicationRequest', pd.DataFrame())
    med_disp_df = pd.concat([_data.get('MimicMedicationDispense', pd.DataFrame()), _data.get('MimicMedicationDispenseED', pd.DataFrame())])
    med_admin_df = pd.concat([_data.get('MimicMedicationAdministration', pd.DataFrame()), _data.get('MimicMedicationAdministrationICU', pd.DataFrame())])
    vitals_df = pd.concat([_data.get('MimicObservationVitalSignsED', pd.DataFrame()), _data.get('MimicObservationChartevents', pd.DataFrame()), _data.get('MimicObservationED', pd.DataFrame()), _data.get('MimicObservationOutputevents', pd.DataFrame()), _data.get('MimicObservationDatetimeevents', pd.DataFrame())])
    labs_df = pd.concat([_data.get('MimicObservationLabevents', pd.DataFrame()), _data.get('MimicObservationMicroSusc', pd.DataFrame()), _data.get('MimicObservationMicroTest', pd.DataFrame()), _data.get('MimicObservationMicroOrg', pd.DataFrame())])
    diag_df = _data.get('MimicDiagnosticReport', pd.DataFrame())
    docs_df = _data.get('MimicDocumentReference', pd.DataFrame())

    for _, enc_row in all_enc_df.iterrows():
        enc_id = enc_row.get('id')
        # get all ICU encounter IDs
        if len(enc_icu_df) > 0:
            icu_enc_data = enc_icu_df[enc_icu_df.get('partOf.reference') == f"Encounter/{enc_id}"]
            icu_enc_ids = icu_enc_data['id'].tolist()
        else:
            icu_enc_data = pd.DataFrame()
            icu_enc_ids = []

        if len(enc_ed_df) > 0:
            ed_enc_data = enc_ed_df[enc_ed_df.get('partOf.reference') == f"Encounter/{enc_id}"]
            ed_enc_ids = ed_enc_data['id'].tolist()
        else:
            ed_enc_data = pd.DataFrame()
            ed_enc_ids = []
        # get all IDs
        all_enc_ids = [enc_id] + icu_enc_ids + ed_enc_ids
        all_enc_ids = [f"Encounter/{id}" for id in all_enc_ids]

        # Related Encounter IDs
        related_enc_ids = []
        if len(icu_enc_data) > 0:
            for _, icu in icu_enc_data.iterrows():
                related_enc_ids.append({
                    "id": icu['id'],
                    "location": "ICU",
                    "start": format_datetime(icu['period.start']),
                    "end": format_datetime(icu['period.end'])
                })
        if len(ed_enc_data) > 0:
            for _, ed in ed_enc_data.iterrows():
                related_enc_ids.append({
                    "id": ed['id'],
                    "location": "ED",
                    "start": format_datetime(ed['period.start']),
                    "end": format_datetime(ed['period.end'])
                })
        

        # --- Conditions ---
        if 'encounter.reference' in all_cond_df.columns:
            enc_conditions_df = all_cond_df[all_cond_df.get('encounter.reference').isin(all_enc_ids)]
        else:
            enc_conditions_df = pd.DataFrame()
        
        # --- Medications ---
        if 'encounter.reference' in med_req_df.columns:
            enc_med_req = med_req_df[med_req_df.get('encounter.reference').isin(all_enc_ids)]
        else:
            enc_med_req = pd.DataFrame()
        
        if 'context.reference' in med_disp_df.columns:
            enc_med_disp = med_disp_df[med_disp_df.get('context.reference').isin(all_enc_ids)]
        else:
            enc_med_disp = pd.DataFrame()
        
        if 'context.reference' in med_admin_df.columns:
            enc_med_admin = med_admin_df[med_admin_df.get('context.reference').isin(all_enc_ids)]
        else:
            enc_med_admin = pd.DataFrame()
        
        meds_req_list = []
        if not enc_med_req.empty:
            for _, row in enc_med_req.iterrows():
                med_name = safe_get(row, ['medicationCodeableConcept', 'coding', 0, 'display'], 'N/A')
                if 'medicationReference.reference' in row and pd.notna(row['medicationReference.reference']):
                    med_id = row['medicationReference.reference'].replace('Medication/', '')
                    med_name = med_map.get(med_id, med_name)

                if med_name != 'N/A':
                    start = safe_get(row, ['dispenseRequest.validityPeriod.start'], 'N/A')
                    if (start != 'N/A') and pd.notna(start):
                        try:
                            start = format_datetime(start)
                        except ValueError:
                            pass
                    end = safe_get(row, ['dispenseRequest.validityPeriod.end'], 'N/A')
                    if (end != 'N/A') and pd.notna(end):
                        try:
                            end = format_datetime(end)
                        except ValueError:
                            pass
                    
                    meds_req_list.append(
                        {
                            'Time': row.get('authoredOn'),
                            'Medication': med_name,
                            'Status': row.get('status'),
                            'Start': start,
                            'End': end,
                            'Dose': safe_get(row, ['dosageInstruction', 0, 'text']),
                            'Route': safe_get(row, ['dosageInstruction', 0, 'route', 'coding', 0, 'code'], 'N/A'),
                        }
                    )

        meds_req_df = pd.DataFrame(meds_req_list)
        if not meds_req_df.empty:
            meds_req_df['Time'] = pd.to_datetime(meds_req_df['Time'], errors='coerce')
            meds_req_df = meds_req_df.dropna(subset=['Time', 'Medication'])
            meds_req_df.sort_values(by=['Medication', 'Time'], inplace=True, ascending=True)
            meds_req_df['Time'] = meds_req_df['Time'].apply(format_datetime)

        meds_disp_list = []
        if not enc_med_disp.empty:
            for _, row in enc_med_disp.iterrows():
                med_name = safe_get(row, ['medicationCodeableConcept', 'coding', 0, 'code'], 'N/A')
                if 'medicationReference.reference' in row and pd.notna(row['medicationReference.reference']):
                    med_id = row['medicationReference.reference'].replace('Medication/', '')
                    med_name = med_map.get(med_id, med_name)

                if med_name != 'N/A':
                    route = safe_get(row, ['dosageInstruction', 0, 'route', 'coding', 0, 'code'], 'N/A')
                    timing = safe_get(row, ['dosageInstruction', 0, 'timing', 'code', 'coding', 0, 'code'], 'N/A')
                    meds_disp_list.append(
                        {
                            'Time': row.get('whenHandedOver', ''),
                            'Medication': med_name,
                            'Status': row.get('status'),
                            'Dose': safe_get(row, ['dosageInstruction', 0, 'text']),
                            'Route': route,
                            'Timing': timing,
                        }
                    )
        meds_disp_df = pd.DataFrame(meds_disp_list)
        if not meds_disp_df.empty:
            meds_disp_df['Time'] = pd.to_datetime(meds_disp_df['Time'], errors='coerce')
            meds_disp_df = meds_disp_df.dropna(subset=['Time', 'Medication'])
            meds_disp_df.sort_values(by=['Medication', 'Time'], inplace=True, ascending=True)
            meds_disp_df['Time'] = meds_disp_df['Time'].apply(format_datetime)
        
        meds_admin_list = []
        if not enc_med_admin.empty:
            for _, row in enc_med_admin.iterrows():
                med_name = safe_get(
                    row,
                    ['medicationCodeableConcept', 'coding', 0, 'display'],
                    safe_get(row, ['medicationCodeableConcept.coding', 0, 'display'], 'N/A')
                )
                dose = format_value(safe_get(
                    row,
                    ['dosage', 'dose', 'value'],
                    safe_get(row, ['dosage.dose.value'], '')
                ))
                unit = safe_get(
                    row,
                    ['dosage', 'dose', 'unit'],
                    safe_get(row, ['dosage.dose.unit'], '')
                )
                route = safe_get(
                    row,
                    ['dosage', 'method', 'coding', 0, 'code'],
                    safe_get(row, ['dosage.method.coding', 0, 'code'], 'N/A')
                )
                if med_name != 'N/A':
                    meds_admin_list.append(
                        {
                            'Time': row.get('effectiveDateTime'), 
                            'Medication': med_name, 
                            'Status': row.get('status'), 
                            'Details': f"{dose} {unit}",
                            'Route': route,
                        }
                    )
        
        meds_admin_df = pd.DataFrame(meds_admin_list)
        if not meds_admin_df.empty:
            meds_admin_df['Time'] = pd.to_datetime(meds_admin_df['Time'], errors='coerce')
            meds_admin_df.sort_values(by=['Medication', 'Time'], inplace=True, ascending=True)
            meds_admin_df['Time'] = meds_admin_df['Time'].apply(format_datetime)

        # --- Vitals ---
        if ('encounter.reference' in vitals_df.columns) and ('context.reference' in vitals_df.columns):
            enc_vitals_df = vitals_df[
                (vitals_df.get('encounter.reference').isin(all_enc_ids)) |
                (vitals_df.get('context.reference').isin(all_enc_ids))
            ]
        elif 'encounter.reference' in vitals_df.columns:
            enc_vitals_df = vitals_df[vitals_df.get('encounter.reference').isin(all_enc_ids)]
        elif 'context.reference' in vitals_df.columns:
            enc_vitals_df = vitals_df[vitals_df.get('context.reference').isin(all_enc_ids)]
        else:
            enc_vitals_df = pd.DataFrame()

        processed_vitals = []   
        if not enc_vitals_df.empty:
            for _, row in enc_vitals_df.iterrows():
                ts = row.get('effectiveDateTime')
                if not ts: continue
                components = row.get('component')
                if isinstance(components, list):
                    for comp in components:
                        vital = safe_get(
                            comp, 
                            ['code', 'coding', 0, 'display'], 
                            safe_get(comp, ['code.coding', 0, 'display'])
                        )
                        vital_group = ''
                        if 'category' in comp:
                            if not pd.isna(comp['category']):
                                vital_group = safe_get(
                                    comp,
                                    ['category', 0, 'coding', 0, 'display'],
                                    safe_get(comp, ['category', 0, 'coding', 0, 'code'])
                                )

                        if (pd.isna(vital_group)) or (vital_group == ''):
                            vital_group = 'Vital Signs' 

                        val = ''
                        if 'valueString' in comp:
                            if not pd.isna(comp['valueString']):
                                val = comp['valueString']
                        if 'valueQuantity' in comp:
                            if not pd.isna(comp['valueQuantity']):
                                val = format_value(comp['valueQuantity']['value'])
                                if 'unit' in comp['valueQuantity']:
                                    if not pd.isna(comp['valueQuantity']['unit']):
                                        vital += f" ({str(comp['valueQuantity']['unit'])})"
                        if 'valueQuantity.value' in comp:
                            if not pd.isna(comp['valueQuantity.value']):
                                val = format_value(comp['valueQuantity.value'])
                                if 'valueQuantity.unit' in comp:
                                    if not pd.isna(comp['valueQuantity.unit']):
                                        vital += f" ({str(comp['valueQuantity.unit'])})"

                        processed_vitals.append(
                            {
                                'Timestamp': ts, 
                                'Vital': vital, 
                                'Vital Group': vital_group,
                                'Value': val
                            })
                else:
                    vital_group = ''
                    if 'category' in row:
                        if not pd.isna(row['category']):
                            vital_group = safe_get(
                                row,
                                ['category', 0, 'coding', 0, 'display'],
                                safe_get(row, ['category', 0, 'coding', 0, 'code'])
                            )

                    if (pd.isna(vital_group)) or (vital_group == ''):
                        vital_group = 'Vital Signs'
                        
                    vital = safe_get(
                        row, 
                        ['code', 'coding', 0, 'display'], 
                        safe_get(row, ['code.coding', 0, 'display'])
                    )
                    val = ''
                    if 'valueString' in row:
                        # check if valueString has a value
                        if not pd.isna(row['valueString']):
                            val = row['valueString']
                    if 'valueQuantity' in row:
                        # check if valueQuantity has a value
                        if not pd.isna(row['valueQuantity']):
                            if 'value' in row['valueQuantity']:
                                if not pd.isna(row['valueQuantity']['value']):
                                    val = format_value(row['valueQuantity']['value'])
                                if 'unit' in row['valueQuantity']:
                                    if not pd.isna(row['valueQuantity']['unit']):
                                        vital += f" ({str(row['valueQuantity']['unit'])})"
                    if 'valueQuantity.value' in row:
                        # check if valueQuantity.value has a value
                        if not pd.isna(row['valueQuantity.value']):
                            val = format_value(row['valueQuantity.value'])
                            if 'valueQuantity.unit' in row:
                                if not pd.isna(row['valueQuantity.unit']):
                                    vital += f" ({str(row['valueQuantity.unit'])})"

                    
                    processed_vitals.append(
                        {
                            'Timestamp': ts, 
                            'Vital': vital, 
                            'Vital Group': vital_group,
                            'Value': val
                        })

        obs_vitals_df = pd.DataFrame(processed_vitals)
        vitals_clean_df = pd.DataFrame()
        observations_clean_df = pd.DataFrame()
        labs_obs_clean_df = pd.DataFrame()

        if not obs_vitals_df.empty:
            # remove survey
            obs_vitals_df = obs_vitals_df[~obs_vitals_df.get('Vital Group').str.lower().str.contains('survey', na=False)]
            # remove called out
            obs_vitals_df = obs_vitals_df[~obs_vitals_df.get('Vital').str.lower().str.contains('called out', na=False)]

            vitals_clean_df = obs_vitals_df[
                (obs_vitals_df.get('Vital Group').str.lower().str.contains('vital', na=False)) |
                (obs_vitals_df.get('Vital Group').str.lower().str.contains('general', na=False))
            ]
            observations_clean_df = obs_vitals_df[
                (~obs_vitals_df.get('Vital Group').str.lower().str.contains('vital', na=False)) & 
                (~obs_vitals_df.get('Vital Group').str.lower().str.contains('general', na=False)) & 
                (obs_vitals_df.get('Vital Group').str.lower() != 'labs')
            ]
            labs_obs_clean_df = obs_vitals_df[
                obs_vitals_df.get('Vital Group').str.lower() == 'labs'
            ]
            if not labs_obs_clean_df.empty:
                labs_obs_clean_df.rename(
                    columns={'Vital': 'Lab Test'}, inplace=True
                )
                labs_obs_clean_df.dropna(subset=['Value'], inplace=True)
                labs_obs_clean_df['Timestamp'] = pd.to_datetime(labs_obs_clean_df['Timestamp'])
                labs_obs_clean_df.drop(columns=['Vital Group'], inplace=True)
                labs_obs_clean_df.sort_values(by='Timestamp', inplace=True, ascending=True)
                labs_obs_clean_df['Timestamp'] = labs_obs_clean_df['Timestamp'].apply(format_datetime)


            if not vitals_clean_df.empty:
                vitals_clean_df.dropna(subset=['Value'], inplace=True)
                vitals_clean_df['Timestamp'] = pd.to_datetime(vitals_clean_df['Timestamp'])
                vitals_clean_df.sort_values(by='Timestamp', inplace=True, ascending=True)
                vitals_clean_df['Timestamp'] = vitals_clean_df['Timestamp'].apply(format_datetime)

            if not observations_clean_df.empty:
                observations_clean_df.dropna(subset=['Value'], inplace=True)
                observations_clean_df.rename(columns={'Vital': 'Observation', 'Vital Group': 'Observation Group'}, inplace=True)
                observations_clean_df['Timestamp'] = pd.to_datetime(observations_clean_df['Timestamp'])
                observations_clean_df.sort_values(by='Timestamp', inplace=True, ascending=True)

                # Precompute adhoc changes and totals
                # Replace JH-HLM
                observations_clean_df['Observation'] = observations_clean_df['Observation'].astype(str).str.replace(
                    'JH-HLM', 'Johns Hopkins Highest Level of Mobility', regex=False
                )

                # Braden Total Score within Skin - Assessment
                try:
                    braden_components = [
                        'Braden Activity',
                        'Braden Friction/Shear',
                        'Braden Mobility',
                        'Braden Moisture',
                        'Braden Nutrition',
                        'Braden Sensory Perception'
                    ]
                    mask_braden = observations_clean_df['Observation Group'] == 'Skin - Assessment'
                    df_braden = observations_clean_df[mask_braden & observations_clean_df['Observation'].isin(braden_components)].copy()
                    if not df_braden.empty:
                        df_braden['__num'] = pd.to_numeric(df_braden['Value'], errors='coerce')
                        braden_totals = df_braden.groupby('Timestamp')['__num'].sum(min_count=1).dropna()
                        if not braden_totals.empty:
                            add_rows = pd.DataFrame({
                                'Observation': 'Braden Total Score',
                                'Observation Group': 'Skin - Assessment',
                                'Timestamp': braden_totals.index,
                                'Value': braden_totals.values
                            })
                            observations_clean_df = pd.concat([observations_clean_df, add_rows], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Braden total score precompute failed: {e}")

                # GCS Total Score within Neurological
                try:
                    gcs_components = [
                        'GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response'
                    ]
                    mask_gcs = observations_clean_df['Observation Group'] == 'Neurological'
                    df_gcs = observations_clean_df[mask_gcs & observations_clean_df['Observation'].isin(gcs_components)].copy()
                    if not df_gcs.empty:
                        df_gcs['__num'] = pd.to_numeric(df_gcs['Value'], errors='coerce')
                        gcs_totals = df_gcs.groupby('Timestamp')['__num'].sum(min_count=1).dropna()
                        if not gcs_totals.empty:
                            add_rows = pd.DataFrame({
                                'Observation': 'GCS - Total Score',
                                'Observation Group': 'Neurological',
                                'Timestamp': gcs_totals.index,
                                'Value': gcs_totals.values
                            })
                            observations_clean_df = pd.concat([observations_clean_df, add_rows], ignore_index=True)
                except Exception as e:
                    logger.warning(f"GCS total score precompute failed: {e}")

                # Final format conversion for UI display
                observations_clean_df['Timestamp'] = observations_clean_df['Timestamp'].apply(format_datetime)

        # --- Labs ---
        if 'encounter.reference' in labs_df.columns:
            enc_labs_df = labs_df[labs_df.get('encounter.reference').isin(all_enc_ids)]
        else:
            enc_labs_df = pd.DataFrame()

        labs_clean_df = pd.DataFrame()
        if not enc_labs_df.empty:
            enc_labs_df['Timestamp'] = pd.to_datetime(enc_labs_df.get('effectiveDateTime'), errors='coerce')
            labs_clean = []
            for _, row in enc_labs_df.iterrows():
                test_name = safe_get(
                    row,
                    ['code.coding', 0, 'display'],
                    safe_get(row, ['code', 'coding', 0, 'display'], 'N/A')
                )
                val = ''
                if 'valueString' in row:
                    if pd.notna(row['valueString']):
                        val = row['valueString']
                if 'valueQuantity' in row:
                    if pd.notna(row['valueQuantity']):
                        val = format_value(row['valueQuantity']['value'])
                        if pd.notna(row['valueQuantity']['unit']):
                            test_name += f" ({row['valueQuantity']['unit']})"
                if 'valueQuantity.value' in row:
                    if pd.notna(row['valueQuantity.value']):
                        val = format_value(row['valueQuantity.value'])
                        if pd.notna(row['valueQuantity.unit']):
                            test_name += f" ({row['valueQuantity.unit']})"

                low_ref = safe_get(
                    row,
                    ['referenceRange', 0, 'low', 'value'],
                    ''
                )
                high_ref = safe_get(
                    row,
                    ['referenceRange', 0, 'high', 'value'],
                    ''
                )

                labs_clean.append({
                    'Timestamp': row['Timestamp'], 
                    'Lab Test': test_name,
                    'Value': val,
                    'Low Ref': low_ref, 
                    'High Ref': high_ref
                })
            
            labs_clean_df = pd.DataFrame(labs_clean)
            if not labs_obs_clean_df.empty:
                labs_clean_df = pd.concat([labs_clean_df, labs_obs_clean_df])

            labs_clean_df.dropna(subset=['Timestamp'], inplace=True)
            labs_clean_df.sort_values(by=['Lab Test', 'Timestamp'], inplace=True, ascending=True)
            labs_clean_df['Timestamp'] = labs_clean_df['Timestamp'].apply(format_datetime)

        # --- Documents ---
        if 'context.encounter' in docs_df.columns:
            docs_df['context.encounter'] = docs_df['context.encounter'].apply(lambda x: x[0]['reference'] if isinstance(x, list) else x)
            enc_docs_df = docs_df[docs_df.get('context.encounter').isin(all_enc_ids)]
        else:
            enc_docs_df = pd.DataFrame()

        if 'encounter.reference' in diag_df.columns:
            diag_df['encounter.reference'] = diag_df['encounter.reference'].apply(lambda x: x[0]['reference'] if isinstance(x, list) else x)
            enc_diag_df = diag_df[diag_df.get('encounter.reference').isin(all_enc_ids)]
        else:
            enc_diag_df = pd.DataFrame()

        # --- Procedures ---
        if 'encounter.reference' in all_proc_df.columns:
            enc_procs_df = all_proc_df[all_proc_df.get('encounter.reference').isin(all_enc_ids)]
        else:
            enc_procs_df = pd.DataFrame()

        enc_procs = []
        for _, row in enc_procs_df.iterrows():
            proc_name = safe_get(
                row, 
                ['code.coding', 0, 'display'], 
                safe_get(row, ['code', 'coding', 0, 'display'], 'N/A')
            )
            proc_code = safe_get(
                row, 
                ['code.coding', 0, 'code'], 
                safe_get(row, ['code', 'coding', 0, 'code'], 'N/A')
            )
            start_time = ''
            if 'performedDateTime' in row:
                if pd.notna(row['performedDateTime']):
                    start_time = row['performedDateTime']
            if 'performedPeriod' in row:
                if pd.notna(row['performedPeriod']) and ('start' in row['performedPeriod']):
                    start_time = row['performedPeriod']['start']
            if 'performedPeriod.start' in row:
                if pd.notna(row['performedPeriod.start']):
                    start_time = row['performedPeriod.start']

            end_time = ''
            if 'performedPeriod' in row:
                if pd.notna(row['performedPeriod']) and ('end' in row['performedPeriod']):
                    end_time = row['performedPeriod']['end']
            if 'performedPeriod.end' in row:
                if pd.notna(row['performedPeriod.end']):
                    end_time = row['performedPeriod.end']

            enc_procs.append({
                'Procedure': proc_name,
                'ProcedureCode': proc_code,
                'StartTime': start_time,
                'EndTime': end_time
            })
        if len(enc_procs) > 0:
            enc_procs_df = pd.DataFrame(enc_procs)
            # enc_procs_df['StartTime'] = pd.to_datetime(enc_procs_df['StartTime'], errors='coerce')
            # enc_procs_df['EndTime'] = pd.to_datetime(enc_procs_df['EndTime'], errors='coerce')
            enc_procs_df['StartTime'] = enc_procs_df['StartTime'].apply(format_datetime)
            enc_procs_df['EndTime'] = enc_procs_df['EndTime'].apply(format_datetime)
        else:
            enc_procs_df = pd.DataFrame()
        
        # Precompute condition groups and lab categories for reuse in UI
        conditions_grouped_df = prepare_grouped_conditions(enc_conditions_df)
        lab_sorted_groups = prepare_lab_sorted_groups(labs_clean_df)

        stitched_data.append({
            **enc_row.to_dict(), 
            'conditions': enc_conditions_df, 
            'conditions_grouped': conditions_grouped_df,
            'procedures': enc_procs_df, 
            'med_request': meds_req_df, 
            'med_disp': meds_disp_df, 
            'med_admin': meds_admin_df, 
            'vitals': vitals_clean_df, 
            'observations': observations_clean_df,
            'labs': labs_clean_df, 
            'lab_sorted_groups': lab_sorted_groups,
            'diagnostic_reports': enc_diag_df,
            'reports': enc_docs_df,
            'related_encounter_ids': related_enc_ids
        })

    final_df = pd.DataFrame(stitched_data)
    # Filter out encounters that are part of a larger one to avoid duplication in the main view
    if 'partOf.reference' in final_df.columns:
        final_df = final_df[final_df['partOf.reference'].isnull()]
    
    return final_df


# --- UI Components ---
def display_welcome_screen():
    """Displays a welcome message when no patient file is loaded."""
    st.title("MIMIC Patient Data Viewer")
    st.markdown("---")
    st.info("Please select or upload a patient's JSON file from the sidebar to begin analysis.")
    st.markdown("""
    This application is designed for healthcare professionals to analyze MIMIC-IV patient data.
    - **Patient Overview**: Demographics, admission timelines, and disease history.
    - **Vitals Dashboard**: Near real-time monitoring and time-series analysis of vital signs.
    - **Lab Results**: Detailed tabular and graphical views of laboratory tests.
    - **Medication Management**: A complete timeline of prescribed and administered drugs.
    - **Procedures and Diagnoses**: Chronological records of interventions and clinical findings.
    """)

def display_patient_overview(patient_data, stitched_enc_df, locations_map, orgs_df, generated_data):
    """Renders the patient overview tab using pre-stitched data."""
    st.header("Patient Overview")

    # --- Patient Demographics ---
    org_map = pd.Series(orgs_df.name.values, index=orgs_df.id).to_dict()
    patient_df = patient_data.get('MimicPatient')
    if patient_df is not None and not patient_df.empty:
        st.subheader("Demographics")
        p_info = patient_df.iloc[0]
        
        birth_date_str = p_info.get('birthDate')
        birth_date_disp = "N/A"
        age = "N/A"
        if birth_date_str:
            try:
                birth_date = datetime.strptime(str(birth_date_str).split('T')[0], '%Y-%m-%d')
                birth_date_disp = format_datetime(birth_date, '%m-%d-%Y')
                age = (datetime.now() - birth_date).days // 365
            except (ValueError, TypeError):
                birth_date_disp = str(birth_date_str)
                age = "Invalid Date"
        
        org_ref = p_info.get('managingOrganization.reference', '').replace('Organization/', '')
        org_name = org_map.get(org_ref, 'Unknown Org')

        # Show fake display name for default patients if provided
        if isinstance(generated_data, dict) and generated_data.get("display_name"):
            st.write(f"Name: {generated_data['display_name']}")
        st.write(f"Patient ID: {patient_data.get('patient_id', 'N/A').split('/')[-1]}")
        st.write(f"Birth Date: {birth_date_disp}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {p_info.get('gender', 'N/A').capitalize()}")
        st.write(f"Race: {safe_get(p_info, ['extension', 0, 'extension', 1, 'valueString'], 'N/A')}")
        st.write(f"Ethnicity: {safe_get(p_info, ['extension', 1, 'extension', 1, 'valueString'], 'N/A')}")
        st.write(f"Marital Status: {safe_get(p_info, ['maritalStatus.coding', 0, 'code'], 'N/A')}")
        st.write(f"Organization: {org_name}")
        st.markdown("---")

    # --- Generated Summaries ---
    if generated_data["summary"] != "":
        with st.expander("AI Generated Patient Summary"):
            st.markdown(generated_data["summary"])
    
    # if generated_data["questions"] != "":
    #     with st.expander("AI Generated Possible Patient Questions"):
    #         st.markdown(generated_data["questions"])

    # --- Encounter Display ---
    st.subheader("Hospital Encounters")
    if stitched_enc_df.empty:
        st.warning("No main hospital admission data available for this patient.")
        return

    for index, enc_row in stitched_enc_df.iterrows():
        start_time = pd.to_datetime(enc_row.get('period.start'), errors='coerce')
        end_time = pd.to_datetime(enc_row.get('period.end'), errors='coerce')
        los = (end_time - start_time).days
        
        enc_conditions_df = enc_row['conditions']
        first_condition = ""
        if not enc_conditions_df.empty:
            first_condition = safe_get(enc_conditions_df.iloc[0]['code.coding'], [0, 'display'], 'N/A')

        enc_class = str(enc_row['_type']).title()

        related_enc_ids = enc_row['related_encounter_ids']
        if pd.notna(start_time):
            start_time = format_datetime(start_time)
        if pd.notna(end_time):
            end_time = format_datetime(end_time)
        else:
            end_time = "Current"
            los = "Current"
        
        expander_title = f"{enc_class}: {start_time} to {end_time} ({los} day(s)) - {first_condition}"

        with st.expander(expander_title):
            st.markdown("#### Admission Details")
            st.write(f"Admission ID: {enc_row.get('id')}")
            st.write(f"Admit Date: {start_time}")
            st.write(f"Discharge Date: {end_time}")
            st.write(f"Length of Stay: {los} day(s)")
            st.write(f"Admit Source: {safe_get(enc_row, ['hospitalization.admitSource.coding', 0, 'code'], 'N/A')}")
            st.write(f"Discharge Disposition: {safe_get(enc_row, ['hospitalization.dischargeDisposition.coding', 0, 'code'], 'N/A')}")

            if len(related_enc_ids) > 0:
                related_enc_ids.sort(key=lambda item:item['start'], reverse=False)
                st.write("Related Encounters:")
                for related_enc in related_enc_ids:
                    st.write(f"- {related_enc['id']} ({related_enc['location']}) : ({related_enc['start']} to {related_enc['end']})")

            if "encounter_summaries" in generated_data:
                if enc_row["id"] in generated_data["encounter_summaries"]:
                    with st.expander("AI Generated Encounter Summary"):
                        st.markdown(generated_data["encounter_summaries"][enc_row["id"]])
                    st.markdown("---")

            st.markdown("#### Further Information")
            # Location Gantt Chart
            with st.expander("Hospital Stay (Locations)"):
                locations = enc_row.get('location')
                if isinstance(locations, list):
                    loc_events = []
                    for loc in locations:
                        loc_id = safe_get(loc, ['location', 'reference'], '').replace('Location/', '')
                        loc_name = locations_map.get(loc_id, 'Unknown Location')

                        start = pd.to_datetime(safe_get(loc, ['period', 'start']), errors='coerce')
                        end = pd.to_datetime(safe_get(loc, ['period', 'end']), errors='coerce')
                        los = (end - start).days
                        loc_events.append({
                            'Location': loc_name,
                            'Start': start,
                            'Finish': end,
                            'Resource': loc_name,
                            'Length of Stay': los
                        })
                    
                    loc_df = pd.DataFrame(loc_events).dropna(subset=['Start', 'Finish'])
                    if not loc_df.empty:
                        fig = px.timeline(
                            loc_df, 
                            x_start="Start", 
                            x_end="Finish", 
                            y="Location", 
                            color="Resource", 
                            custom_data="Length of Stay",
                            title=f"Patient Movement for Encounter {enc_row.get('id')}"
                        )
                        fig.update_yaxes(autorange="reversed")
                        fig.update_layout( 
                            showlegend=False
                        )
                        # fig.update_traces(
                        #     hovertemplate="<b>Location:</b> %{y}<br>" +
                        #             "<b>Start:</b> %{x|%Y-%m-%d %H:%M:%S}<br>" +
                        #             "<b>End:</b> %{xother|%Y-%m-%d %H:%M:%S}<br>" +
                        #             "<b>Length of Stay:</b> %{customdata[0]} day(s)"
                        # )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("No location data for this encounter.")
                else:
                    st.write("No location data for this encounter.")

            # Nested Expanders
            with st.expander("Conditions"):
                if not enc_conditions_df.empty:
                    # Prefer precomputed grouping from stitch_encounter_data
                    cond_grouped = enc_row.get('conditions_grouped')
                    if isinstance(cond_grouped, pd.DataFrame) and not cond_grouped.empty:
                        cond_df = cond_grouped.copy()
                    else:
                        cond_df = prepare_grouped_conditions(enc_conditions_df)

                    for group in sorted(cond_df['Group'].unique()):
                        st.subheader(group.title())
                        st.dataframe(
                            cond_df[cond_df['Group'] == group].drop('Group', axis=1),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.write("No condition data for this encounter.")

            with st.expander("Procedures"):
                enc_procs_df = enc_row['procedures']
                if not enc_procs_df.empty:
                    enc_procs_df.sort_values('StartTime', inplace=True)
                    st.dataframe(enc_procs_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No procedure data for this encounter.")

            with st.expander("Medications"):
                meds_req_df = enc_row['med_request']
                meds_disp_df = enc_row['med_disp']
                meds_admin_df = enc_row['med_admin']
                
                if not meds_req_df.empty:
                    st.subheader("Medication Orders")
                    st.dataframe(meds_req_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
                
                if not meds_disp_df.empty:
                    st.subheader("Medication Dispensed")
                    st.dataframe(meds_disp_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
                
                if not meds_admin_df.empty:
                    st.subheader("Medication Administrations")
                    st.dataframe(meds_admin_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
                
                if meds_req_df.empty and meds_disp_df.empty and meds_admin_df.empty:
                    st.write("No medication data for this encounter.")

            with st.expander("Vitals"):
                vitals_clean_df = enc_row['vitals']
                vitals_clean_df.drop_duplicates(['Vital', 'Vital Group', 'Timestamp'], inplace=True)
                
                if not vitals_clean_df.empty:
                    # Group vitals into clinically-meaningful categories and display per group
                    # Prepare base pivoted data for easy slicing by Vital name
                    vitals_clean_df.sort_values('Timestamp', ascending=True, inplace=True)
                    vit_pivot = vitals_clean_df.pivot(index='Vital', columns='Timestamp', values='Value')
                    vit_pivot.reset_index(inplace=True)

                    # Pre-index pivot rows by vital name for fast selection
                    name_to_row = {row['Vital']: row for _, row in vit_pivot.iterrows()}

                    # Build grouped lists using unique Vital names
                    unique_vital_names = sorted(vitals_clean_df['Vital'].unique().tolist())
                    # Bucket by category
                    buckets: Dict[str, List[str]] = {}
                    for name in unique_vital_names:
                        cat = classify_vital(name)
                        buckets.setdefault(cat, []).append(name)

                    # Render groups by priority
                    for category in sorted(buckets.keys(), key=vital_category_priority):
                        ordered_names = order_names_within_category(category, buckets[category])
                        if not ordered_names:
                            continue
                        st.subheader(category)
                        # Reconstruct a sub-table for just these vitals in the requested order
                        sub_df = pd.DataFrame([name_to_row[n] for n in ordered_names if n in name_to_row])
                        sub_df.dropna(how='all', axis=1, inplace=True)
                        sub_df.fillna(value="", inplace=True)
                        st.dataframe(sub_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No vital signs data for this encounter.")

            with st.expander("Observations (Chart events / Flowsheets)"):
                obs_clean_df = enc_row['observations']
                obs_clean_df.drop_duplicates(['Observation', 'Observation Group', 'Timestamp'], inplace=True)

                
                if not obs_clean_df.empty:
                    for group in obs_clean_df['Observation Group'].sort_values(ascending=True).unique():
                        st.subheader(group)
                        obs_clean_df_group = obs_clean_df[obs_clean_df['Observation Group'] == group]
                        obs_clean_df_group.sort_values('Timestamp', ascending=True, inplace=True)
                        obs_clean_df_group_pivot = obs_clean_df_group.pivot(index='Observation', columns='Timestamp', values='Value')
                        obs_clean_df_group_pivot.reset_index(inplace=True)
                        obs_clean_df_group_pivot.dropna(how='all', axis=1, inplace=True)
                        obs_clean_df_group_pivot.fillna(value="", inplace=True)

                        # Note: JH-HLM replacement and total score calculations are precomputed
                        # upstream in stitch_encounter_data for consistency and performance.
                        # This UI now renders the precomputed results.
                        obs_clean_df_group_pivot.sort_values('Observation', inplace=True)
                        st.dataframe(obs_clean_df_group_pivot, use_container_width=True, hide_index=True)
                else:
                    st.write("No observation data for this encounter.")

            with st.expander("Labs"):
                labs_clean_df = enc_row['labs']
                labs_clean_df.drop_duplicates(['Lab Test', 'Timestamp'], inplace=True)
                if not labs_clean_df.empty:
                    labs_clean_df_pivot = labs_clean_df.pivot(index='Lab Test', columns='Timestamp', values='Value')
                    labs_clean_df_pivot.reset_index(inplace=True)

                    # Prefer precomputed lab categories from stitch_encounter_data
                    pre_sorted = enc_row.get('lab_sorted_groups')
                    if isinstance(pre_sorted, dict) and pre_sorted:
                        for group, labs in pre_sorted.items():
                            st.subheader(group.replace("_", " "))
                            labs_group = labs_clean_df_pivot[labs_clean_df_pivot['Lab Test'].isin(labs)].dropna(axis=1, how='all')
                            labs_group.fillna(value="", inplace=True)
                            st.dataframe(labs_group, use_container_width=True, hide_index=True)
                    else:
                        # Fallback to inline sorting using functional API
                        sorted_labs = sort_labs_by_category(labs_clean_df_pivot['Lab Test'].unique().tolist())
                        for group, labs in sorted_labs.items():
                            st.subheader(group.replace("_", " "))
                            labs_group = labs_clean_df_pivot[labs_clean_df_pivot['Lab Test'].isin(labs)].dropna(axis=1, how='all')
                            labs_group.fillna(value="", inplace=True)
                            st.dataframe(labs_group, use_container_width=True, hide_index=True)
                else:
                    st.write("No lab data for this encounter.")

            with st.expander("Diagnostic Reports"):
                enc_diag_df = enc_row['diagnostic_reports']
                if not enc_diag_df.empty:
                    enc_diag_df.sort_values('effectiveDateTime', inplace=True)
                    for _, row in enc_diag_df.iterrows():
                        doc_title = safe_get(row, ['presentedForm', 0, 'title'], "Document")
                        doc_date = row.get('effectiveDateTime', 'No Date')
                        with st.expander(f"**{doc_title} - {format_datetime(doc_date, '%m-%d-%Y %H:%M:%S')}**"):
                            try:
                                b64_data = safe_get(row, ['presentedForm', 0, 'data'])
                                if b64_data:
                                    text_data = base64.b64decode(b64_data).decode('UTF-8', errors='ignore')
                                    st.text_area(f"{row['id']}", text_data, height=300)
                                else:
                                    st.warning("No data found for this document.")
                            except Exception as e:
                                st.error(f"Could not display document. Error: {e}")
                else:
                    st.write("No reports for this encounter.")

            with st.expander("Clinical Documents"):
                enc_docs_df = enc_row['reports']
                if not enc_docs_df.empty:
                    enc_docs_df.sort_values('date', inplace=True)
                    for _, row in enc_docs_df.iterrows():
                        doc_title = safe_get(row, ['content', 0, 'attachment', 'title'], "Document")
                        doc_date = row.get('date', 'No Date')
                        with st.expander(f"**{doc_title} - {format_datetime(doc_date, '%m-%d-%Y %H:%M:%S')}**"):
                            try:
                                b64_data = safe_get(row, ['content', 0, 'attachment', 'data'])
                                if b64_data:
                                    text_data = base64.b64decode(b64_data).decode('UTF-8', errors='ignore')
                                    st.text_area(f"{row['id']}", text_data, height=300)
                                else:
                                    st.warning("No data found for this document.")
                            except Exception as e:
                                st.error(f"Could not display document. Error: {e}")
                else:
                    st.write("No reports for this encounter.")

def display_vitals_dashboard(stitched_enc_df: pd.DataFrame) -> None:
    """Render the vital signs dashboard with table/graph toggle."""
    st.header("Vital Signs Dashboard")

    vitals_df = pd.DataFrame()
    for _, enc_row in stitched_enc_df.iterrows():
        vitals_clean_df = enc_row['vitals']
        vitals_clean_df.drop_duplicates(['Vital', 'Timestamp'], inplace=True)
        vitals_df = pd.concat([vitals_df, vitals_clean_df])

    if vitals_df.empty:
        st.warning("No vital signs data available for this patient.")
        return

    # st.subheader("Latest Readings")
    # # st.dataframe(vitals_df)
    # hr_val = get_latest_vital(vitals_df, 'Heart Rate')
    # sbp_val = get_latest_vital(vitals_df, 'Non Invasive Blood Pressure systolic')
    # dbp_val = get_latest_vital(vitals_df, 'Non Invasive Blood Pressure diastolic')
    # rr_val = get_latest_vital(vitals_df, 'Respiratory Rate')
    # temp_val = get_latest_vital(vitals_df, 'Temperature Fahrenheit')
    # o2_val = get_latest_vital(vitals_df, 'O2 saturation pulseoxymetry')

    # cols = st.columns(3)
    # cols2 = st.columns(3)
    # cols[0].metric(label="❤️ Heart Rate", value=hr_val)
    # cols[1].metric(label="🩸 Systolic BP", value=sbp_val)
    # cols[2].metric(label="🩸 Diastolic BP", value=dbp_val)
    # cols2[0].metric(label="💨 Resp. Rate", value=rr_val)
    # cols2[1].metric(label="🌡️ Temp (F)", value=temp_val)
    # cols2[2].metric(label="💨 SpO2", value=o2_val)

    # st.markdown("---")
    
    st.subheader("Vital Signs Over Time")
    # vitals_df['Vital_label'] = vitals_df['Vital Group'] + " - " + vitals_df['Vital']
    unique_vitals = vitals_df['Vital'].sort_values(ascending=True).unique()
    default_vitals = [
        "Heart Rate", 
        "Non Invasive Blood Pressure systolic", 
        "Non Invasive Blood Pressure diastolic", 
        "Respiratory Rate", 
        "Temperature Fahrenheit", 
        "O2 saturation pulseoxymetry",
        "Heart rate",
        "Respiratory rate",
        "Body temperature",
        "Systolic blood pressure",
        "Diastolic blood pressure"
    ]

    default_vitals = [g for g in default_vitals if g in unique_vitals]
    selected_vitals = st.multiselect(
        "Select vitals to display:", 
        options=unique_vitals, 
        default=default_vitals
    )
    # add time slider
    vitals_df['Timestamp'] = pd.to_datetime(vitals_df['Timestamp'])
    time_slider = st.slider(
        "Time Slider", 
        min_value=vitals_df['Timestamp'].min().date(),
        max_value=vitals_df['Timestamp'].max().date()+pd.Timedelta(days=1),
        value=(vitals_df['Timestamp'].min().date(), vitals_df['Timestamp'].max().date()+pd.Timedelta(days=1)),
        key="vitals_time_slider"
    )
    
    if selected_vitals:
        # Filter selection and time window
        selected = vitals_df[
            (vitals_df['Vital'].isin(selected_vitals)) &
            (vitals_df['Timestamp'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ].copy()

        # View toggle
        view = st.radio("View", ["Graph", "Table"], index=1, horizontal=True, key="vitals_view")

        if view == "Graph":
            vitals_to_plot = selected.copy()
            vitals_to_plot['NumericValue'] = pd.to_numeric(
                vitals_to_plot['Value'].astype(str).str.extract(r'(\d*\.?\d+)')[0],
                errors='coerce'
            )
            vitals_to_plot = vitals_to_plot.dropna(subset=['NumericValue'])

            selected_vitals_to_plot = vitals_to_plot['Vital'].unique().tolist()
            n_facets = max(1, len(selected_vitals_to_plot))
            fig = make_subplots(
                rows=n_facets,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=selected_vitals_to_plot if selected_vitals_to_plot else ["No data"]
            )

            for i, vital in enumerate(selected_vitals_to_plot, start=1):
                vital_data = vitals_to_plot[vitals_to_plot['Vital'] == vital]
                hover_timestamps = vital_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                units = vital_data['Unit'] if 'Unit' in vital_data.columns else [''] * len(vital_data)
                hover_template = (
                    '<b>%{customdata[0]}</b><br>'
                    'Time: %{customdata[1]}<br>'
                    'Value: %{y:.2f} %{customdata[2]}<br>'
                    '<extra></extra>'
                )
                fig.add_trace(
                    go.Scatter(
                        x=vital_data['Timestamp'],
                        y=vital_data['NumericValue'],
                        mode='lines+markers',
                        name=vital,
                        line=dict(color=f'rgb({i * 50 % 255}, {i * 100 % 255}, {i * 150 % 255})'),
                        customdata=list(zip(vital_data['Vital'], hover_timestamps, units)),
                        hovertemplate=hover_template
                    ),
                    row=i,
                    col=1
                )

            fig.update_layout(height=200 * n_facets, showlegend=False, title_text="Vital Signs Trend", title_x=0.5, margin=dict(t=60))
            for i in range(1, n_facets + 1):
                fig.update_yaxes(title_text="Reading", row=i, col=1, title_standoff=10)
            fig.update_xaxes(title_text="Time", row=n_facets, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            table = selected.sort_values(['Vital', 'Timestamp']).pivot_table(index='Timestamp', columns='Vital', values='Value', aggfunc='first')
            st.dataframe(table, use_container_width=True)
        

def display_labs_dashboard(stitched_enc_df: pd.DataFrame) -> None:
    """Render the laboratory results with table/graph toggle and microbiology table."""
    st.header("Laboratory Results")
    labs_df = pd.DataFrame()
    for _, enc_row in stitched_enc_df.iterrows():
        labs_clean_df = enc_row['labs']
        labs_clean_df.drop_duplicates(['Lab Test', 'Timestamp'], inplace=True)
        labs_df = pd.concat([labs_df, labs_clean_df])
    
    if labs_df.empty:
        st.warning("No laboratory data available for this patient.")
        return

    st.subheader("Lab Results Over Time")
    unique_labs = labs_df['Lab Test'].unique()
    default_labs = [
        "Hemoglobin", 
        "Glucose", 
        "Creatinine", 
        "Bilirubin", 
        "Albumin", 
        "Platelets",
        "Potassium",
        "White Blood Cells",
        "Red Blood Cells"
    ]

    default_labs = [v for v in default_labs if v in unique_labs]
    selected_labs = st.multiselect(
        "Select labs to display:", 
        options=unique_labs, 
        default=default_labs
    )
    # add time slider
    labs_df['Timestamp'] = pd.to_datetime(labs_df['Timestamp'])
    time_slider = st.slider(
        "Time Slider", 
        min_value=labs_df['Timestamp'].min().date(), 
        max_value=labs_df['Timestamp'].max().date()+pd.Timedelta(days=1), 
        value=(labs_df['Timestamp'].min().date(), labs_df['Timestamp'].max().date()+pd.Timedelta(days=1)),
        key="labs_time_slider"
    )
    
    if selected_labs:
        selected = labs_df[
            (labs_df['Lab Test'].isin(selected_labs)) &
            (labs_df['Timestamp'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ].copy()

        view = st.radio("View", ["Graph", "Table"], index=1, horizontal=True, key="labs_view")

        if view == "Graph":
            labs_to_plot = selected.copy()
            labs_to_plot['NumericValue'] = pd.to_numeric(
                labs_to_plot['Value'].astype(str).str.extract(r'(\d*\.?\d+)')[0],
                errors='coerce'
            )
            labs_to_plot = labs_to_plot.dropna(subset=['NumericValue'])

            selected_labs_to_plot = labs_to_plot['Lab Test'].unique().tolist()
            n_facets = max(1, len(selected_labs_to_plot))
            fig = make_subplots(
                rows=n_facets,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=selected_labs_to_plot if selected_labs_to_plot else ["No data"]
            )

            for i, lab in enumerate(selected_labs_to_plot, start=1):
                lab_data = labs_to_plot[labs_to_plot['Lab Test'] == lab]
                hover_timestamps = lab_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                units = lab_data['Unit'] if 'Unit' in lab_data.columns else [''] * len(lab_data)
                hover_template = (
                    '<b>%{customdata[0]}</b><br>'
                    'Time: %{customdata[1]}<br>'
                    'Value: %{y:.2f} %{customdata[2]}<br>'
                    '<extra></extra>'
                )
                fig.add_trace(
                    go.Scatter(
                        x=lab_data['Timestamp'],
                        y=lab_data['NumericValue'],
                        mode='lines+markers',
                        name=lab,
                        line=dict(color=f'rgb({i * 50 % 255}, {i * 100 % 255}, {i * 150 % 255})'),
                        customdata=list(zip(lab_data['Lab Test'], hover_timestamps, units)),
                        hovertemplate=hover_template
                    ),
                    row=i,
                    col=1
                )

            fig.update_layout(height=200 * n_facets, showlegend=False, title_text="Lab Results Trend", title_x=0.5, margin=dict(t=60))
            for i in range(1, n_facets + 1):
                fig.update_yaxes(title_text="Reading", row=i, col=1, title_standoff=10)
            fig.update_xaxes(title_text="Time", row=n_facets, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            table = selected.sort_values(['Lab Test', 'Timestamp']).pivot_table(index='Timestamp', columns='Lab Test', values='Value', aggfunc='first')
            st.dataframe(table, use_container_width=True)


def display_medications(stitched_enc_df):
    """Renders the medication management tab."""
    st.header("Medication Management")

    med_req_df = pd.DataFrame()
    med_admin_df = pd.DataFrame()
    med_disp_df = pd.DataFrame()

    for _, enc_row in stitched_enc_df.iterrows():
        med_req_df = pd.concat([med_req_df, enc_row['med_request']])
        med_admin_df = pd.concat([med_admin_df, enc_row['med_admin']])
        med_disp_df = pd.concat([med_disp_df, enc_row['med_disp']])
    
    # get all medications
    med_df = pd.concat([med_req_df, med_admin_df, med_disp_df])
    if med_df.empty:
        st.write("No medication data for this encounter.")
        return

    all_meds = med_df['Medication'].unique()

    # search for medications
    selected_meds = st.multiselect(
        "Select medications to display:", 
        options=all_meds,
        default=all_meds
    )
    # add time slider
    med_df['Time'] = pd.to_datetime(med_df['Time'])
    
    time_slider = st.slider(
        "Time Slider", 
        min_value=med_df['Time'].min().date(), 
        max_value=med_df['Time'].max().date()+pd.Timedelta(days=1), 
        value=(med_df['Time'].min().date(), med_df['Time'].max().date()+pd.Timedelta(days=1)),
        key="med_time_slider"
    )

    if not med_req_df.empty:
        st.subheader("Medication Orders")
        med_req_df['Time'] = pd.to_datetime(med_req_df['Time'])
        med_req_df = med_req_df[
            (med_req_df['Medication'].isin(selected_meds)) & 
            (med_req_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        med_req_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(med_req_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
    
    if not med_disp_df.empty:
        st.subheader("Medication Dispensed")
        med_disp_df['Time'] = pd.to_datetime(med_disp_df['Time'])
        med_disp_df = med_disp_df[
            (med_disp_df['Medication'].isin(selected_meds)) & 
            (med_disp_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        med_disp_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(med_disp_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
    
    if not med_admin_df.empty:
        st.subheader("Medication Administrations")
        med_admin_df['Time'] = pd.to_datetime(med_admin_df['Time'])
        med_admin_df = med_admin_df[
            (med_admin_df['Medication'].isin(selected_meds)) & 
            (med_admin_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        med_admin_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(med_admin_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)


def display_procedures(stitched_enc_df):
    """Renders the procedures and interventions tab."""
    st.header("Procedures and Interventions")

    all_procedures = []
    for _, enc_row in stitched_enc_df.iterrows():
        all_procedures.append(enc_row['procedures'])
    
    all_procedures_df = pd.concat(all_procedures)
    if all_procedures_df.empty:
        st.write("No procedure data for this encounter.")
        return
    
    # add time slider
    all_procedures_df['StartTime'] = pd.to_datetime(all_procedures_df['StartTime'])
    time_slider = st.slider(
        "Time Slider", 
        min_value=all_procedures_df['StartTime'].min().date(), 
        max_value=all_procedures_df['StartTime'].max().date()+pd.Timedelta(days=1), 
        value=(all_procedures_df['StartTime'].min().date(), all_procedures_df['StartTime'].max().date()+pd.Timedelta(days=1)),
        key="proc_time_slider"
    )

    all_procedures_df = all_procedures_df[
        (all_procedures_df['StartTime'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
    ]
    all_procedures_df.sort_values('StartTime', ascending=True, inplace=True)
    st.dataframe(all_procedures_df, use_container_width=True, hide_index=True)
    

def display_documents(stitched_enc_df):
    """Renders clinical documents like discharge summaries."""
    st.header("Clinical Documents")

    all_documents = []
    for _, enc_row in stitched_enc_df.iterrows():
        all_documents.append(enc_row['reports'])
        all_documents.append(enc_row['diagnostic_reports'])
    
    all_documents_df = pd.concat(all_documents)
    if all_documents_df.empty:
        st.write("No document data for this encounter.")
        return

    all_documents_df['date'].fillna(all_documents_df['effectiveDateTime'], inplace=True)
    all_documents_df['date'] = pd.to_datetime(all_documents_df['date'])

    # add time slider
    all_documents_df['date'] = pd.to_datetime(all_documents_df['date'])
    time_slider = st.slider(
        "Time Slider", 
        min_value=all_documents_df['date'].min().date(), 
        max_value=all_documents_df['date'].max().date()+pd.Timedelta(days=1), 
        value=(all_documents_df['date'].min().date(), all_documents_df['date'].max().date()+pd.Timedelta(days=1)),
        key="doc_time_slider"
    )
    
    all_documents_df['_type'] = all_documents_df['type.text'].fillna(all_documents_df['code.text'])
    doc_types = (
        all_documents_df['_type'].dropna().unique().tolist()
    )
    doc_types = list(set(doc_types))
    document_type = st.multiselect(
        "Select document type", 
        options=doc_types,
        default=doc_types
    )

    all_documents_df = all_documents_df[
        (all_documents_df['date'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1]))) & 
        (all_documents_df['_type'].isin(document_type))
    ]

    all_documents_df.sort_values('date', inplace=True)
    for _, row in all_documents_df.iterrows():
        doc_title = safe_get(
            row, ['content', 0, 'attachment', 'title'], 
            safe_get(row, ['presentedForm', 0, 'title'], "Document")
        )
        doc_date = row.get('date')
        with st.expander(f"**{doc_title} - {format_datetime(doc_date, '%m-%d-%Y %H:%M:%S')}**"):
            try:
                b64_data = safe_get(
                    row, ['content', 0, 'attachment', 'data'], 
                    safe_get(row, ['presentedForm', 0, 'data'])
                )
                if b64_data:
                    text_data = base64.b64decode(b64_data).decode('UTF-8', errors='ignore')
                    st.text_area(f"{row['id']}", text_data, height=300, key=row['id'])
                else:
                    st.warning("No data found for this document.")
            except Exception as e:
                st.error(f"Could not display document. Error: {e}")


# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app."""
    # Load reference data from assets
    locations_df = load_ndjson_data("assets/reference_data/MimicLocation.ndjson")
    medications_df = load_ndjson_data("assets/reference_data/MimicMedication.ndjson")
    # specimens_df = load_ndjson_data("assets/reference_data/MimicSpecimen.ndjson")
    orgs_df = load_ndjson_data("assets/reference_data/MimicOrganization.ndjson")

    with open("assets/patient_summaries.json", "r") as f:
        patient_summaries = json.load(f)

    with st.sidebar:
        st.title("👨‍⚕️ MIMIC Patient Viewer")
        # Default patient selector
        default_dir = "assets/mimic_default_patients"
        default_patients = list_default_patients(default_dir)
        default_labels = []
        default_paths = []
        for path, pid in default_patients:
            default_paths.append(path)
            display_id = pid[-8:]
            default_labels.append(f"{pseudonymize_patient_id(pid)} — {display_id}")
        selected_default_idx = 1
        if default_labels:
            # Default to the 2nd index (index=1) when available
            default_index = 0
            selected_default_idx = st.selectbox(
                "Select a default patient",
                options=list(range(len(default_labels))),
                index=default_index,
                format_func=lambda i: default_labels[i],
            )

        st.markdown("Or upload your own JSON file")
        uploaded_file = st.file_uploader("Upload patient JSON", type=['json'])

    selected_default_path = default_paths[selected_default_idx] if (default_labels and selected_default_idx is not None) else None

    if uploaded_file:
        patient_data = load_patient_data(uploaded_file)
        display_name = None
    elif selected_default_path:
        patient_data = load_patient_data_file(selected_default_path)
        # derive patient_id from filename for pseudonym
        try:
            pid_from_name = selected_default_path.split('/')[-1].replace('patient_', '').replace('.json', '')
            display_name = pseudonymize_patient_id(pid_from_name)
        except Exception:
            display_name = None
    
    if patient_data:
        st.title("Patient Dashboard")
        
        # --- Prepare data maps and stitch encounters ---
        locations_map = pd.Series(locations_df.name.values, index=locations_df.id).to_dict()
        
        def get_med_name(identifiers):
            if isinstance(identifiers, list):
                for ident in identifiers:
                    if isinstance(ident, dict) and 'system' in ident and 'mimic-medication-name' in ident['system']:
                        return ident.get('value', 'Unknown Med')
            return 'Unknown Med'
        medications_df['display_name'] = medications_df['identifier'].apply(get_med_name)
        med_map = pd.Series(medications_df.display_name.values, index=medications_df.id).to_dict()

        stitched_encounters_df = stitch_encounter_data(patient_data, locations_map, med_map)
        
        # tab_titles = ["📄 Overview", "❤️ Vitals", "🧪 Labs", "💊 Medications", "💉 Procedures", "📝 Documents"]
        # overview, vitals, labs, medications, procedures, documents = st.tabs(tab_titles)

        patient_id = patient_data["patient_id"].replace("Patient/", "")
        generated_data = {
            "summary": "",
            "questions": "",
            "encounter_summaries": {}
        }
        if patient_id in patient_summaries:
            generated_data = patient_summaries[patient_id]
        # Attach a display name if this is a default patient selection
        if display_name:
            generated_data["display_name"] = display_name

        # with overview:
        display_patient_overview(patient_data, stitched_encounters_df, locations_map, orgs_df, generated_data)
        # with vitals:
        #     display_vitals_dashboard(stitched_encounters_df)
        # with labs:
        #     display_labs_dashboard(stitched_encounters_df)
        # with medications:
        #     display_medications(stitched_encounters_df)
        # with procedures:
        #     display_procedures(stitched_encounters_df)
        # with documents:
        #     display_documents(stitched_encounters_df)
    else:
        display_welcome_screen()

if __name__ == "__main__":
    main()
