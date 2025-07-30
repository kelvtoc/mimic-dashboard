import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- Configuration & Constants ---
st.set_page_config(page_title="MIMIC Patient Data Viewer", layout="wide", initial_sidebar_state="expanded")


# --- Data Loading and Caching ---
@st.cache_data
def load_ndjson_data(file):
    """Loads a newline-delimited JSON string into a DataFrame."""
    with open(file, 'r') as f:
        file_content_string = f.read()
    lines = file_content_string.strip().split('\n')
    records = [json.loads(line) for line in lines]
    return pd.json_normalize(records)

@st.cache_data
def load_patient_data(uploaded_file):
    """Loads and parses the uploaded JSON patient file into a dictionary of DataFrames."""
    if uploaded_file is None:
        return None
    
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(file_content)
        
        patient_id = data.get('patient_id', 'Unknown Patient')
        fhir_data = data.get('data', {})
        
        processed_data = {'patient_id': patient_id}
        
        for resource_type, records in fhir_data.items():
            if records:
                processed_data[resource_type] = pd.json_normalize(records, max_level=3)
            else:
                 processed_data[resource_type] = pd.DataFrame()

        return processed_data
    except Exception as e:
        st.error(f"Error loading or parsing file: {e}")
        return None

# --- Helper Functions ---
def safe_get(dct, keys, default=None):
    """Safely get a nested value from a dictionary."""
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, TypeError, IndexError):
            return default
    return dct

def format_value(val):
    try:
        # Convert to float and check if it's an integer
        num = float(val)
        if num.is_integer():
            return str(int(num))  # Return "1" for 1.0
        return f"{num:.2f}"  # Return 2 decimal places for non-integer floats
    except (ValueError, TypeError):
        return str(val)  # Return unchanged if string or invalid

def get_display_name(row, key_list):
    """Safely extracts display name from common FHIR structures."""
    display = safe_get(row, key_list)
    if display:
        return display
    return row.get(key_list[0], 'N/A')

def format_datetime(series, format_str='%Y-%m-%d %H:%M:%S'):
    """Safely converts a series to datetime and formats it."""
    return pd.to_datetime(series, errors='coerce').strftime(format_str)

def get_latest_vital(df, vital_name):
    """Gets the most recent value for a specific vital sign."""
    latest = df[df['Vital'] == vital_name].sort_values(by='Timestamp', ascending=False)
    if len(latest) > 0:
        latest = latest.iloc[0]
        value = latest['Value']
        return value
    return "N/A"

def style_lab_results(df):
    """Applies color coding to lab results based on reference ranges."""
    def highlight_abnormal(row):
        style = ''
        try:
            low = float(row.get('Low Ref'))
            high = float(row.get('High Ref'))
            value = float(row.get('Value'))
            
            if pd.notnull(value):
                if pd.notnull(low) and value < low:
                    style = 'background-color: #FFCCCC'
                elif pd.notnull(high) and value > high:
                    style = 'background-color: #FFCCCC'
        except (ValueError, TypeError):
            pass 
            
        return [style] * len(row)

    return df.style.apply(highlight_abnormal, axis=1)

# --- Data Stitching Logic ---
# @st.cache_data
def stitch_encounter_data(_data, locations_map, med_map):
    """Stitches all related patient data into a single encounters DataFrame."""
    
    # Combine all encounter types
    enc_df = _data.get('MimicEncounter', pd.DataFrame())
    enc_ed_df = _data.get('MimicEncounterED', pd.DataFrame())
    enc_icu_df = _data.get('MimicEncounterICU', pd.DataFrame())
    
    all_enc_df = pd.concat([enc_df, enc_ed_df], ignore_index=True)
    if all_enc_df.empty:
        return pd.DataFrame()
        
    all_enc_df.sort_values(by='period.start', inplace=True)
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
    micro_org_df = pd.concat([_data.get('MimicObservationMicroSusc', pd.DataFrame()), _data.get('MimicObservationMicroTest', pd.DataFrame()), _data.get('MimicObservationMicroOrg', pd.DataFrame())])
    labs_df = _data.get('MimicObservationLabevents', pd.DataFrame())
    docs_df = _data.get('MimicDocumentReference', pd.DataFrame())

    for _, enc_row in all_enc_df.iterrows():
        enc_id = enc_row.get('id')
        # get all ICU encounter IDs
        if len(enc_icu_df) > 0:
            icu_enc_ids = enc_icu_df[enc_icu_df.get('partOf.reference') == f"Encounter/{enc_id}"]
            icu_enc_ids = icu_enc_ids['id'].tolist()
        else:
            icu_enc_ids = []

        if len(enc_ed_df) > 0:
            ed_enc_ids = enc_ed_df[enc_ed_df.get('partOf.reference') == f"Encounter/{enc_id}"]
            ed_enc_ids = ed_enc_ids['id'].tolist()
        else:
            ed_enc_ids = []
        # get all IDs
        all_enc_ids = [enc_id] + icu_enc_ids + ed_enc_ids
        all_enc_ids = [f"Encounter/{id}" for id in all_enc_ids]

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
                            start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                    end = safe_get(row, ['dispenseRequest.validityPeriod.end'], 'N/A')
                    if (end != 'N/A') and pd.notna(end):
                        try:
                            end = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                    
                    meds_req_list.append(
                        {
                            'Time': row.get('authoredOn'),
                            'Medication': med_name,
                            'Status': row.get('status'),
                            'Period': f"{start} - {end}",
                            'Dose': safe_get(row, ['dosageInstruction', 0, 'text']),
                            'Route': safe_get(row, ['dosageInstruction', 0, 'route', 'coding', 0, 'code'], 'N/A'),
                        }
                    )

        meds_req_df = pd.DataFrame(meds_req_list)
        if not meds_req_df.empty:
            meds_req_df['Time'] = pd.to_datetime(meds_req_df['Time'], errors='coerce')
            meds_req_df = meds_req_df.dropna(subset=['Time', 'Medication'])
            meds_req_df.sort_values(by=['Medication', 'Time'], inplace=True, ascending=True)

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
                        val = ''
                        if 'valueString' in comp:
                            if not pd.isna(comp['valueString']):
                                val = comp['valueString']
                        if 'valueQuantity' in comp:
                            if not pd.isna(comp['valueQuantity']):
                                val = format_value(comp['valueQuantity']['value'])
                                if 'unit' in comp['valueQuantity']:
                                    if not pd.isna(comp['valueQuantity']['unit']):
                                        val += str(comp['valueQuantity']['unit'])
                        if 'valueQuantity.value' in comp:
                            if not pd.isna(comp['valueQuantity.value']):
                                val = format_value(comp['valueQuantity.value'])
                                if 'valueQuantity.unit' in comp:
                                    if not pd.isna(comp['valueQuantity.unit']):
                                        val += str(comp['valueQuantity.unit'])
                        
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

                        processed_vitals.append(
                            {
                                'Timestamp': ts, 
                                'Vital': vital, 
                                'Vital Group': vital_group,
                                'Value': val
                            })
                else:
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
                                        val += str(row['valueQuantity']['unit'])
                    if 'valueQuantity.value' in row:
                        # check if valueQuantity.value has a value
                        if not pd.isna(row['valueQuantity.value']):
                            val = format_value(row['valueQuantity.value'])
                            if 'valueQuantity.unit' in row:
                                if not pd.isna(row['valueQuantity.unit']):
                                    val += str(row['valueQuantity.unit'])

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
            vitals_clean_df = obs_vitals_df[obs_vitals_df.get('Vital Group').str.lower().str.contains('vital', na=False)]
            observations_clean_df = obs_vitals_df[
                (~obs_vitals_df.get('Vital Group').str.lower().str.contains('vital', na=False)) & 
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


            if not vitals_clean_df.empty:
                vitals_clean_df.dropna(subset=['Value'], inplace=True)
                vitals_clean_df['Timestamp'] = pd.to_datetime(vitals_clean_df['Timestamp'])
                vitals_clean_df.sort_values(by='Timestamp', inplace=True, ascending=True)

            if not observations_clean_df.empty:
                observations_clean_df.dropna(subset=['Value'], inplace=True)
                observations_clean_df.rename(columns={'Vital': 'Observation', 'Vital Group': 'Observation Group'}, inplace=True)
                observations_clean_df['Timestamp'] = pd.to_datetime(observations_clean_df['Timestamp'])
                observations_clean_df.sort_values(by='Timestamp', inplace=True, ascending=True)

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
                            val += str(row['valueQuantity']['unit'])
                if 'valueQuantity.value' in row:
                    if pd.notna(row['valueQuantity.value']):
                        val = format_value(row['valueQuantity.value'])
                        if pd.notna(row['valueQuantity.unit']):
                            val += str(row['valueQuantity.unit'])

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

        # --- Documents ---
        if 'context.encounter' in docs_df.columns:
            docs_df['context.encounter'] = docs_df['context.encounter'].apply(lambda x: x[0]['reference'] if isinstance(x, list) else x)
            enc_docs_df = docs_df[docs_df.get('context.encounter').isin(all_enc_ids)]
        else:
            enc_docs_df = pd.DataFrame()

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
            enc_procs_df['StartTime'] = pd.to_datetime(enc_procs_df['StartTime'], errors='coerce')
            enc_procs_df['EndTime'] = pd.to_datetime(enc_procs_df['EndTime'], errors='coerce')
        else:
            enc_procs_df = pd.DataFrame()


        # --- Microbiology ---
        enc_micro = []
        for _, row in micro_org_df.iterrows():
            micro_name = safe_get(
                row, 
                ['code.coding', 0, 'display'], 
                safe_get(row, ['code', 'coding', 0, 'display'], 'N/A')
            )

            val = ''
            if 'valueString' in row:
                if pd.notna(row['valueString']):
                    val = row['valueString']

            if 'valueCodeableConcept.coding' in row:
                if pd.notna(row['valueCodeableConcept.coding']):
                    val = row['valueCodeableConcept.coding'][0]['display']

            time = ''
            if 'effectiveDateTime' in row:
                if pd.notna(row['effectiveDateTime']):
                    time = row['effectiveDateTime']

            enc_micro.append({
                'Microbiology': micro_name,
                'Value': val,
                'Time': time
            })
        if len(enc_micro) > 0:
            enc_micro_df = pd.DataFrame(enc_micro)
            enc_micro_df['Time'] = pd.to_datetime(enc_micro_df['Time'], errors='coerce')
        else:
            enc_micro_df = pd.DataFrame()
        
        
        stitched_data.append({
            **enc_row.to_dict(), 
            'conditions': enc_conditions_df, 
            'procedures': enc_procs_df, 
            'med_request': meds_req_df, 
            'med_disp': meds_disp_df, 
            'med_admin': meds_admin_df, 
            'vitals': vitals_clean_df, 
            'observations': observations_clean_df,
            'labs': labs_clean_df, 
            'microorg': enc_micro_df,
            'reports': enc_docs_df
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

def display_patient_overview(patient_data, stitched_enc_df, locations_map, orgs_df):
    """Renders the patient overview tab using pre-stitched data."""
    st.header("Patient Overview")

    # --- Patient Demographics ---
    org_map = pd.Series(orgs_df.name.values, index=orgs_df.id).to_dict()
    patient_df = patient_data.get('MimicPatient')
    if patient_df is not None and not patient_df.empty:
        st.subheader("Demographics")
        p_info = patient_df.iloc[0]
        
        birth_date_str = p_info.get('birthDate')
        age = "N/A"
        if birth_date_str:
            try:
                birth_date = datetime.strptime(birth_date_str.split('T')[0], '%Y-%m-%d')
                age = (datetime.now() - birth_date).days // 365
            except (ValueError, TypeError):
                age = "Invalid Date"
        
        org_ref = p_info.get('managingOrganization.reference', '').replace('Organization/', '')
        org_name = org_map.get(org_ref, 'Unknown Org')

        st.write(f"Patient ID: {patient_data.get('patient_id', 'N/A').split('/')[-1]}")
        st.write(f"Birth Date: {birth_date.strftime('%Y-%m-%d')}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {p_info.get('gender', 'N/A').capitalize()}")
        st.write(f"Race: {safe_get(p_info, ['extension', 0, 'extension', 1, 'valueString'], 'N/A')}")
        st.write(f"Ethnicity: {safe_get(p_info, ['extension', 1, 'extension', 1, 'valueString'], 'N/A')}")
        st.write(f"Marital Status: {safe_get(p_info, ['maritalStatus.coding', 0, 'code'], 'N/A')}")
        st.write(f"Organization: {org_name}")
        st.markdown("---")

    # --- Encounter Display ---
    st.subheader("Admissions")
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

        enc_class = str(safe_get(enc_row, ['class.display'], 'Admission')).title()

        expander_title = f"{enc_class}: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({los} day(s)) - {first_condition}"

        with st.expander(expander_title):
            st.markdown("#### Admission Details")
            st.write(f"Admission ID: {enc_row.get('id')}")
            st.write(f"Admit Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Discharge Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Length of Stay: {los} day(s)")
            st.write(f"Admit Source: {safe_get(enc_row, ['hospitalization.admitSource.coding', 0, 'code'], 'N/A')}")
            st.write(f"Discharge Disposition: {safe_get(enc_row, ['hospitalization.dischargeDisposition.coding', 0, 'code'], 'N/A')}")

            # Location Gantt Chart
            with st.expander("Location"):
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
                            'Task': loc_name,  # Plotly expects 'Task' for the y-axis label
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
                            y="Task", 
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
                    cond_list = [
                        safe_get(row['code.coding'], [0, 'display'], 'N/A') 
                        for _, row in enc_conditions_df.iterrows()
                    ]
                    code_list = [
                        safe_get(row['code.coding'], [0, 'code'], 'N/A') 
                        for _, row in enc_conditions_df.iterrows()
                    ]
                    cond_df = pd.DataFrame({'Condition': cond_list, 'Code': code_list})
                    st.dataframe(cond_df, use_container_width=True, hide_index=True)
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
                    st.subheader("Medication Requests")
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
                    # for group in vitals_clean_df['Vital Group'].sort_values(ascending=True).unique():
                    #     st.subheader(group)
                    # vitals_clean_df_group = vitals_clean_df[vitals_clean_df['Vital Group'] == group]
                    vitals_clean_df.sort_values('Timestamp', ascending=True, inplace=True)
                    vitals_clean_df_pivot = vitals_clean_df.pivot(index='Vital', columns='Timestamp', values='Value')
                    vitals_clean_df_pivot.reset_index(inplace=True)
                    vitals_clean_df_pivot.fillna(value="", inplace=True)
                    st.dataframe(vitals_clean_df_pivot, use_container_width=True, hide_index=True)
                else:
                    st.write("No vital signs data for this encounter.")

            with st.expander("Observations"):
                obs_clean_df = enc_row['observations']
                obs_clean_df.drop_duplicates(['Observation', 'Observation Group', 'Timestamp'], inplace=True)

                
                if not obs_clean_df.empty:
                    for group in obs_clean_df['Observation Group'].sort_values(ascending=True).unique():
                        st.subheader(group)
                        obs_clean_df_group = obs_clean_df[obs_clean_df['Observation Group'] == group]
                        obs_clean_df_group.sort_values('Timestamp', ascending=True, inplace=True)
                        obs_clean_df_group_pivot = obs_clean_df_group.pivot(index='Observation', columns='Timestamp', values='Value')
                        obs_clean_df_group_pivot.reset_index(inplace=True)
                        obs_clean_df_group_pivot.fillna(value="", inplace=True)
                        st.dataframe(obs_clean_df_group_pivot, use_container_width=True, hide_index=True)
                else:
                    st.write("No observation data for this encounter.")

            with st.expander("Labs"):
                labs_clean_df = enc_row['labs']
                labs_clean_df.drop_duplicates(['Lab Test', 'Timestamp'], inplace=True)
                if not labs_clean_df.empty:
                    labs_clean_df_pivot = labs_clean_df.pivot(index='Lab Test', columns='Timestamp', values='Value')
                    labs_clean_df_pivot.reset_index(inplace=True)
                    labs_clean_df_pivot.fillna(value="", inplace=True)
                    st.dataframe(labs_clean_df_pivot, use_container_width=True, hide_index=True)
                else:
                    st.write("No lab data for this encounter.")

                st.subheader("Microbiology")
                microorg_df = enc_row['microorg']
                if not microorg_df.empty:
                    st.dataframe(microorg_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No microbiology data for this encounter.")

            with st.expander("Reports"):
                enc_docs_df = enc_row['reports']
                if not enc_docs_df.empty:
                    enc_docs_df.sort_values('date', inplace=True)
                    for _, row in enc_docs_df.iterrows():
                        doc_title = safe_get(row, ['content', 0, 'attachment', 'title'], "Document")
                        doc_date = row.get('date', 'No Date')
                        with st.expander(f"**{doc_title} - {format_datetime(doc_date, '%Y-%m-%d')}**"):
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

def display_vitals_dashboard(stitched_enc_df):
    """Renders the vital signs dashboard."""
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
    time_slider = st.slider(
        "Time Slider", 
        min_value=vitals_df['Timestamp'].min().date(),
        max_value=vitals_df['Timestamp'].max().date()+pd.Timedelta(days=1),
        value=(vitals_df['Timestamp'].min().date(), vitals_df['Timestamp'].max().date()+pd.Timedelta(days=1)),
        key="vitals_time_slider"
    )
    
    if selected_vitals:
        vitals_to_plot = vitals_df[
            (vitals_df['Vital'].isin(selected_vitals)) & 
            (vitals_df['Timestamp'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        vitals_to_table = vitals_to_plot.copy()
        vitals_to_plot['Value'] = pd.to_numeric(
            vitals_to_plot['Value'].str.extract(r'(\d*\.?\d+)')[0],
            errors='coerce'
        )
        vitals_to_plot = vitals_to_plot.dropna(subset=['Value'])

        # Create subplot figure with one row per vital sign
        selected_vitals_to_plot = vitals_to_plot['Vital'].unique().tolist()
        n_facets = len(selected_vitals_to_plot)
        fig = make_subplots(
            rows=n_facets,
            cols=1,
            shared_xaxes=True,  # Share x-axis (Timestamp) across subplots
            vertical_spacing=0.05,  # Adjust spacing between subplots
            subplot_titles=selected_vitals_to_plot  # Set subplot titles to lab test names
        )

        # Add traces with custom hover template
        for i, vital in enumerate(selected_vitals_to_plot, start=1):
            vital_data = vitals_to_plot[vitals_to_plot['Vital'] == vital]
            
            # Format Timestamp for hover (assuming Timestamp is datetime)
            hover_timestamps = vital_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Include Unit in hover if available, otherwise use empty string
            units = vital_data['Unit'] if 'Unit' in vital_data.columns else [''] * len(vital_data)
            
            # Custom hover template
            hover_template = (
                '<b>%{customdata[0]}</b><br>'  # Lab Test name
                'Time: %{customdata[1]}<br>'   # Formatted Timestamp
                'Value: %{y:.2f} %{customdata[2]}<br>'  # Value with unit
                '<extra></extra>'  # Removes secondary box
            )
            
            fig.add_trace(
                go.Scatter(
                    x=vital_data['Timestamp'],
                    y=vital_data['Value'],
                    mode='lines+markers',
                    name=vital,
                    line=dict(color=f'rgb({i * 50 % 255}, {i * 100 % 255}, {i * 150 % 255})'),
                    customdata=list(zip(vital_data['Vital'], hover_timestamps, units)),  # Pass data for hover
                    hovertemplate=hover_template
                ),
                row=i,
                col=1
            )

        # Update layout
        fig.update_layout(
            height=200 * n_facets,
            showlegend=False,
            title_text="Vital Signs Trend",
            title_x=0.5,
            title_y=0.98,
            margin=dict(t=100)
        )

        # Customize axes
        for i in range(1, n_facets + 1):
            fig.update_yaxes(title_text="Reading", row=i, col=1, title_standoff=10)
        fig.update_xaxes(title_text="Time", row=n_facets, col=1)

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        vitals_to_table = vitals_to_table[
            ~vitals_to_table['Vital'].isin(vitals_to_plot['Vital'].unique().tolist())
        ]
        if not vitals_to_table.empty:
            vitals_to_table.sort_values(['Vital', 'Timestamp'], inplace=True)
            st.dataframe(vitals_to_table, use_container_width=True, hide_index=True)
        

def display_labs_dashboard(stitched_enc_df):
    """Renders the laboratory results tab."""
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
    time_slider = st.slider(
        "Time Slider", 
        min_value=labs_df['Timestamp'].min().date(), 
        max_value=labs_df['Timestamp'].max().date()+pd.Timedelta(days=1), 
        value=(labs_df['Timestamp'].min().date(), labs_df['Timestamp'].max().date()+pd.Timedelta(days=1)),
        key="labs_time_slider"
    )
    
    if selected_labs:
        labs_to_plot = labs_df[
            (labs_df['Lab Test'].isin(selected_labs)) & 
            (labs_df['Timestamp'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        labs_to_table = labs_to_plot.copy()
        labs_to_plot['Value'] = pd.to_numeric(
            labs_to_plot['Value'].str.extract(r'(\d*\.?\d+)')[0],
            errors='coerce'
        )
        labs_to_plot = labs_to_plot.dropna(subset=['Value'])

        # Create subplot figure with one row per lab test
        selected_labs_to_plot = labs_to_plot['Lab Test'].unique()
        n_facets = len(selected_labs_to_plot)
        fig = make_subplots(
            rows=n_facets,
            cols=1,
            shared_xaxes=True,  # Share x-axis (Timestamp) across subplots
            vertical_spacing=0.05,  # Adjust spacing between subplots
            subplot_titles=selected_labs_to_plot  # Set subplot titles to lab test names
        )

        # Add traces with custom hover template
        for i, lab in enumerate(selected_labs_to_plot, start=1):
            lab_data = labs_to_plot[labs_to_plot['Lab Test'] == lab]
            
            # Format Timestamp for hover (assuming Timestamp is datetime)
            hover_timestamps = lab_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Include Unit in hover if available, otherwise use empty string
            units = lab_data['Unit'] if 'Unit' in lab_data.columns else [''] * len(lab_data)
            
            # Custom hover template
            hover_template = (
                '<b>%{customdata[0]}</b><br>'  # Lab Test name
                'Time: %{customdata[1]}<br>'   # Formatted Timestamp
                'Value: %{y:.2f} %{customdata[2]}<br>'  # Value with unit
                '<extra></extra>'  # Removes secondary box
            )
            
            fig.add_trace(
                go.Scatter(
                    x=lab_data['Timestamp'],
                    y=lab_data['Value'],
                    mode='lines+markers',
                    name=lab,
                    line=dict(color=f'rgb({i * 50 % 255}, {i * 100 % 255}, {i * 150 % 255})'),
                    customdata=list(zip(lab_data['Lab Test'], hover_timestamps, units)),  # Pass data for hover
                    hovertemplate=hover_template
                ),
                row=i,
                col=1
            )

        # Update layout
        fig.update_layout(
            height=200 * n_facets,
            showlegend=False,
            title_text="Lab Results Trend",
            title_x=0.5,
            title_y=0.98,
            margin=dict(t=100)
        )

        # Customize axes
        for i in range(1, n_facets + 1):
            fig.update_yaxes(title_text="Reading", row=i, col=1, title_standoff=10)
        fig.update_xaxes(title_text="Time", row=n_facets, col=1)

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        labs_to_table = labs_to_table[
            ~labs_to_table['Lab Test'].isin(labs_to_plot['Lab Test'].unique().tolist())
        ]
        if not labs_to_table.empty:
            labs_to_table.sort_values(['Lab Test', 'Timestamp'], inplace=True)
            st.dataframe(labs_to_table, use_container_width=True, hide_index=True)

    st.header("Microbiology")
    microorg_df = enc_row['microorg']
    if not microorg_df.empty:
        microorg_df = microorg_df[
            microorg_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1]))
        ]
        microorg_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(microorg_df, use_container_width=True, hide_index=True)
    else:
        st.write("No microbiology data for this encounter.")


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
    time_slider = st.slider(
        "Time Slider", 
        min_value=med_df['Time'].min().date(), 
        max_value=med_df['Time'].max().date()+pd.Timedelta(days=1), 
        value=(med_df['Time'].min().date(), med_df['Time'].max().date()+pd.Timedelta(days=1)),
        key="med_time_slider"
    )

    if not med_req_df.empty:
        st.subheader("Medication Requests")
        med_req_df = med_req_df[
            (med_req_df['Medication'].isin(selected_meds)) & 
            (med_req_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        med_req_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(med_req_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
    
    if not med_disp_df.empty:
        st.subheader("Medication Dispensed")
        med_disp_df = med_disp_df[
            (med_disp_df['Medication'].isin(selected_meds)) & 
            (med_disp_df['Time'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
        ]
        med_disp_df.sort_values('Time', ascending=True, inplace=True)
        st.dataframe(med_disp_df.dropna(subset=['Time']), use_container_width=True, hide_index=True)
    
    if not med_admin_df.empty:
        st.subheader("Medication Administrations")
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
    
    all_documents_df = pd.concat(all_documents)
    if all_documents_df.empty:
        st.write("No document data for this encounter.")
        return

    all_documents_df['date'] = pd.to_datetime(all_documents_df['date'])

    # add time slider
    time_slider = st.slider(
        "Time Slider", 
        min_value=all_documents_df['date'].min().date(), 
        max_value=all_documents_df['date'].max().date()+pd.Timedelta(days=1), 
        value=(all_documents_df['date'].min().date(), all_documents_df['date'].max().date()+pd.Timedelta(days=1)),
        key="doc_time_slider"
    )
    
    all_documents_df = all_documents_df[
        (all_documents_df['date'].between(pd.Timestamp(time_slider[0]), pd.Timestamp(time_slider[1])))
    ]
    all_documents_df.sort_values('date', inplace=True)
    for _, row in all_documents_df.iterrows():
        doc_title = safe_get(row, ['content', 0, 'attachment', 'title'], "Document")
        doc_date = row.get('date', 'No Date')
        with st.expander(f"**{doc_title} - {format_datetime(doc_date, '%Y-%m-%d')}**"):
            try:
                b64_data = safe_get(row, ['content', 0, 'attachment', 'data'])
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
    # Load reference data
    locations_df = load_ndjson_data("data/mimic_assets/MimicLocation.ndjson")
    medications_df = load_ndjson_data("data/mimic_assets/MimicMedication.ndjson")
    # specimens_df = load_ndjson_data("data/mimic_assets/MimicSpecimen.ndjson")
    orgs_df = load_ndjson_data("data/mimic_assets/MimicOrganization.ndjson")

    with st.sidebar:
        st.title("👨‍⚕️ MIMIC Patient Viewer")
        uploaded_file = st.file_uploader("Choose a patient JSON file", type=['json'])

    if uploaded_file:
        patient_data = load_patient_data(uploaded_file)
        
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
            
            tab_titles = ["📄 Overview", "❤️ Vitals", "🧪 Labs", "💊 Medications", "💉 Procedures", "📝 Documents"]
            overview, vitals, labs, medications, procedures, documents = st.tabs(tab_titles)
            
            with overview:
                display_patient_overview(patient_data, stitched_encounters_df, locations_map, orgs_df)
            with vitals:
                display_vitals_dashboard(stitched_encounters_df)
            with labs:
                display_labs_dashboard(stitched_encounters_df)
            with medications:
                display_medications(stitched_encounters_df)
            with procedures:
                display_procedures(stitched_encounters_df)
            with documents:
                display_documents(stitched_encounters_df)
    else:
        display_welcome_screen()

if __name__ == "__main__":
    main()
