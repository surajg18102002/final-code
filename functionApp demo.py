import azure.functions as func
import logging
import json
import os
import io
import re
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.blob import BlobServiceClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

_df_cache = None

# ============================================================
# SECRETS — all from Azure Function Environment Variables
# ============================================================

SNOW_GET_URL     = "https://apigateway-crt.caresource.corp/ServiceNow/Incident/1.0"
SNOW_GET_KEY     = os.environ["SNOW_GET_KEY"]

TIDAL_URL        = os.environ["TIDAL_BASE_URL"]
TIDAL_AUTH_TOKEN = os.environ["TIDAL_AUTH_TOKEN"]   # pre-encoded base64 token, used as-is
TIDAL_API_KEY    = os.environ["TIDAL_API_KEY"]

SNOW_UPDATE_URL  = "https://apigateway-crt.caresource.corp/ServiceNow/Incident/CreateUpdate/Sync/"
SNOW_UPDATE_KEY  = os.environ["SNOW_UPDATE_KEY"]

GPT_ENDPOINT     = os.environ["GPT_ENDPOINT"]
GPT_KEY          = os.environ["GPT_KEY"]
GPT_MODEL        = os.environ["GPT_MODEL"]


def get_tidal_headers() -> dict:
    """Tidal auth headers — lowercase 'basic' to match working Postman call."""
    return {
        "User-Agent":    "CareSource-TidalJobs/1.2",
        "Authorization": f"basic {TIDAL_AUTH_TOKEN}",   # lowercase — matches working Postman
        "x-api-key":     TIDAL_API_KEY,
        "Content-Type":  "application/x-www-form-urlencoded"
    }


# ============================================================
# KNOWLEDGE BASE HELPERS (unchanged)
# ============================================================

def load_knowledge_base():
    global _df_cache
    if _df_cache is not None:
        logging.info("Using cached knowledge base.")
        return _df_cache

    logging.info("Loading knowledge base from blob storage...")

    conn_str  = os.environ["BLOB_CONNECTION_STRING"]
    container = os.environ["BLOB_CONTAINER_NAME"]
    filename  = os.environ["BLOB_FILE_NAME"]

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client  = blob_service.get_blob_client(container=container, blob=filename)
    blob_data    = blob_client.download_blob().readall()

    try:
        df = pd.read_csv(io.BytesIO(blob_data), encoding='utf-8')
    except UnicodeDecodeError:
        logging.warning("UTF-8 failed, trying latin-1 encoding")
        df = pd.read_csv(io.BytesIO(blob_data), encoding='latin-1')

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Number':             'number',
        'Configuration item': 'config_item',
        'Work notes':         'work_notes',
        'Resolution notes':   'resolution_notes'
    })

    df['work_notes']       = df['work_notes'].fillna('')
    df['resolution_notes'] = df['resolution_notes'].fillna('')
    df['config_item']      = df['config_item'].fillna('')

    df['search_text'] = df.apply(
        lambda r: build_search_text(r['work_notes'], r['resolution_notes'], r['config_item']),
        axis=1
    )
    df = df[df['search_text'].str.strip() != ''].reset_index(drop=True)

    _df_cache = df
    logging.info(f"Loaded {len(df)} records, {df['config_item'].nunique()} config items.")
    return df


def build_search_text(work_notes: str, resolution_notes: str, config_item: str) -> str:
    combined = work_notes + ' ' + resolution_notes
    combined = re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [AP]M - .+?\(Work notes\)', ' ', combined)
    for phrase in [
        'Task auto-reassigned to Managed Services team',
        'Work notes from INCTASK',
        'Please investigate',
        'Job Output can be found in Tidal',
        'Please see work notes for parent group and agent information',
    ]:
        combined = combined.replace(phrase, ' ')
    combined = re.sub(r'\s+', ' ', combined).strip()
    return f"{config_item} {combined}"


def extract_resolution_action(work_notes: str, resolution_notes: str) -> str:
    combined = work_notes + '\n' + resolution_notes
    combined = re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [AP]M - .+?\(Work notes\)', '\n', combined)
    for phrase in [
        'Task auto-reassigned to Managed Services team',
        'Work notes from INCTASK',
        'Please investigate',
        'Job Output can be found in Tidal',
        'Please see work notes for parent group and agent information',
    ]:
        combined = combined.replace(phrase, '')
    combined = re.sub(r'\n+', ' ', combined)
    combined = re.sub(r'\s+', ' ', combined).strip()
    return combined if combined else 'No resolution details available'


def find_similar_tickets(df, config_item: str, query_text: str, top_n: int = 3):
    filtered = df[df['config_item'].str.strip() == config_item.strip()].copy()
    logging.info(f"Stage 1 [{config_item}]: {len(filtered)} tickets")

    if len(filtered) < 15:
        logging.info("< 15 results — expanding to full dataset")
        filtered = df.copy()

    if len(filtered) == 0:
        return []

    corpus            = filtered['search_text'].tolist()
    corpus_with_query = corpus + [query_text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_features=8000
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_with_query)
    except Exception as e:
        logging.error(f"TF-IDF error: {e}")
        return []

    query_vector  = tfidf_matrix[-1]
    corpus_matrix = tfidf_matrix[:-1]
    similarities  = cosine_similarity(query_vector, corpus_matrix).flatten()

    filtered = filtered.copy()
    filtered['similarity'] = similarities
    top_results = filtered.sort_values('similarity', ascending=False).head(top_n)

    results = []
    for _, row in top_results.iterrows():
        results.append({
            "ticket_number":     row['number'],
            "config_item":       row['config_item'],
            "resolution_action": extract_resolution_action(
                row['work_notes'], row['resolution_notes']
            )[:800],
            "similarity_score":  round(float(row['similarity']), 4)
        })
    return results


def clean_for_snow(text: str) -> str:
    """Clean GPT output for safe ServiceNow work note posting."""
    # Remove or replace characters that break Snow API
    text = text.replace('"', "'")
    text = text.replace('\\', '/')
    text = re.sub(r'[^\x00-\x7F]', '', text)   # strip non-ASCII unicode
    text = text.strip()
    return text[:3000]                           # cap at 3000 chars


# ============================================================
# ROUTE 1 — existing (unchanged)
# ============================================================

@app.route(route="find-similar-tickets", methods=["POST", "GET"])
def find_similar_tickets_http(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("find-similar-tickets triggered.")

    job_name      = req.params.get("job_name", "")
    job_id        = req.params.get("job_id", "")
    ticket_number = req.params.get("ticket_number", "")

    if not job_name:
        try:
            body          = req.get_json()
            job_name      = body.get("job_name", "").strip()
            job_id        = body.get("job_id", "").strip()
            ticket_number = body.get("ticket_number", "").strip()
        except Exception:
            pass

    job_name      = job_name.strip()
    job_id        = job_id.strip()
    ticket_number = ticket_number.strip()

    if not job_name:
        return func.HttpResponse(
            json.dumps({"error": "job_name is required"}),
            status_code=400,
            mimetype="application/json"
        )

    config_item = job_name
    query_text  = f"{job_name} {config_item} Completed Abnormally"

    try:
        df    = load_knowledge_base()
        top_3 = find_similar_tickets(df, config_item, query_text, top_n=3)
    except Exception as e:
        logging.error(f"Search error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

    return func.HttpResponse(
        json.dumps({
            "ticket_number": ticket_number,
            "job_name":      job_name,
            "top_3_similar": top_3
        }, indent=2),
        status_code=200,
        mimetype="application/json"
    )


# ============================================================
# ROUTE 2 — full pipeline
# ============================================================

@app.route(route="process-incident", methods=["GET", "POST"])
def process_incident(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("process-incident triggered")

    # ── 1. Get incident number ────────────────────────────────────────────
    incident_number = (
        req.params.get("incident_number")
        or (req.get_json(silent=True) or {}).get("incident_number", "")
    )
    incident_number = incident_number.strip().upper()
    if not incident_number:
        return func.HttpResponse(
            json.dumps({"error": "incident_number is required"}),
            status_code=400,
            mimetype="application/json"
        )
    if not incident_number.startswith("INC"):
        incident_number = "INC" + incident_number

    pipeline = {"incident_number": incident_number, "steps": {}}

    # ── 2. Fetch ServiceNow ───────────────────────────────────────────────
    try:
        snow_resp = requests.get(
            SNOW_GET_URL,
            params={"IncidentNumber": incident_number, "Source": "Streamline"},
            headers={"X-API-KEY": SNOW_GET_KEY, "Accept": "application/json"},
            timeout=30,
            verify=False
        )
        snow_resp.raise_for_status()
        snow_data = snow_resp.json()
        pipeline["steps"]["servicenow_fetch"] = "success"
    except Exception as e:
        pipeline["steps"]["servicenow_fetch"] = f"FAILED: {e}"
        return func.HttpResponse(
            json.dumps(pipeline),
            status_code=500,
            mimetype="application/json"
        )

    incident_details = (
        snow_data.get("ResponseData", {})
                 .get("IncidentDetails", [{}])[0]
    )
    short_desc  = incident_details.get("ShortDescription", "")
    config_item = incident_details.get("ConfigurationItem", "")
    work_notes  = incident_details.get("Activity", {}).get("WorkNotes", [""])[0]

    parts    = [p.strip() for p in short_desc.split(" | ")]
    job_name = parts[0].replace("PRD Tidal Job ", "").strip() if parts else ""
    job_id   = parts[1] if len(parts) > 1 else ""

    pipeline["parsed"] = {
        "job_name":    job_name,
        "job_id":      job_id,
        "short_desc":  short_desc,
        "config_item": config_item
    }

    # ── 3. Fetch Tidal log ────────────────────────────────────────────────
    tidal_output = ""
    if job_id:
        try:
            xml_payload = (
                f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<entry xmlns="http://purl.org/atom/ns#">'
                f'<tes:JobOutput.getJobOutputRaw xmlns:tes="http://www.tidalsoftware.com/client/tesservlet">'
                f'<id>{job_id}</id>'
                f'</tes:JobOutput.getJobOutputRaw>'
                f'</entry>'
            )
            tidal_resp = requests.post(
                TIDAL_URL,
                headers=get_tidal_headers(),
                data={"data": xml_payload},
                timeout=60,
                verify=False
            )
            tidal_resp.raise_for_status()
            tidal_output = tidal_resp.text
            pipeline["steps"]["tidal_fetch"] = "success"
        except Exception as e:
            tidal_output = f"Tidal fetch failed: {e}"
            pipeline["steps"]["tidal_fetch"] = f"FAILED: {e}"
    else:
        pipeline["steps"]["tidal_fetch"] = "skipped — no job_id parsed"

    # ── 4. Similarity search ──────────────────────────────────────────────
    try:
        df         = load_knowledge_base()
        query_text = f"{job_name} {config_item} Completed Abnormally {tidal_output[:500]}"
        top_3      = find_similar_tickets(df, job_name, query_text, top_n=3)
        pipeline["steps"]["similarity_search"] = "success"
    except Exception as e:
        top_3 = []
        pipeline["steps"]["similarity_search"] = f"FAILED: {e}"

    similar_text = "\n".join([
        f"{i+1}. Ticket {t['ticket_number']} (score: {t['similarity_score']}): {t['resolution_action']}"
        for i, t in enumerate(top_3)
    ]) or "No similar tickets found"

    # ── 5. GPT analysis ───────────────────────────────────────────────────
    final_report = ""
    try:
        gpt_client = ChatCompletionsClient(
            endpoint=GPT_ENDPOINT,
            credential=AzureKeyCredential(GPT_KEY)
        )

        system_prompt = (
            "You are a senior IT Operations analyst at CareSource specializing in Tidal "
            "Enterprise Scheduler job failures. Your job is to analyze failed Tidal jobs "
            "and produce clear, structured work notes for ServiceNow incidents. "
            "You write in plain text only — no markdown, no bullet symbols, no asterisks. "
            "You are concise, technical, and actionable. Every recommendation must be specific "
            "and directly tied to the evidence from the error log and past incidents."
        )

        user_prompt = (
            f"INCIDENT DETAILS\n"
            f"Ticket Number : {incident_number}\n"
            f"Job Name      : {job_name}\n"
            f"Job Run ID    : {job_id}\n"
            f"Configuration : {config_item}\n\n"

            f"TIDAL ERROR LOG\n"
            f"{tidal_output[:2000] if tidal_output and 'failed' not in tidal_output.lower() else 'Not available - Tidal fetch failed'}\n\n"

            f"TOP 3 SIMILAR PAST INCIDENTS (ranked by similarity)\n"
            f"{similar_text}\n\n"

            f"INSTRUCTIONS\n"
            f"Using the above information write a ServiceNow work note with these 4 sections:\n\n"
            f"SECTION 1 - JOB FAILURE SUMMARY\n"
            f"State clearly which job failed, its run ID, and what the Tidal log shows as the error. "
            f"Be specific — include any error codes, exit codes, or failure messages from the log. "
            f"2 to 3 sentences maximum.\n\n"
            f"SECTION 2 - ERROR MESSAGE\n"
            f"Extract and quote the exact error message or failure reason from the Tidal log. "
            f"If the log is unavailable, state that and describe the failure based on the incident details.\n\n"
            f"SECTION 3 - RELATED PAST INCIDENTS\n"
            f"List all 3 similar incidents. For each one state the ticket number, "
            f"what happened in that incident, and how it was resolved. "
            f"Note the similarity score to indicate how closely it matches the current issue.\n\n"
            f"SECTION 4 - RECOMMENDED RESOLUTION STEPS\n"
            f"Based on the error log and how similar incidents were resolved, "
            f"provide 4 to 6 numbered steps that the engineer should follow to resolve this incident. "
            f"Steps must be specific and actionable — not generic advice. "
            f"Reference the past incidents where relevant.\n\n"
            f"Plain text only. No markdown formatting. No bullet points or asterisks."
        )

        gpt_response = gpt_client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
            model=GPT_MODEL,
            max_tokens=1000,
            temperature=0.2
        )
        final_report = gpt_response.choices[0].message.content
        pipeline["steps"]["gpt_analysis"] = "success"
    except Exception as e:
        final_report = f"GPT analysis failed: {e}"
        pipeline["steps"]["gpt_analysis"] = f"FAILED: {e}"

    # ── 6. Update ServiceNow (add work note only) ─────────────────────────
    clean_report = clean_for_snow(final_report)
    try:
        update_resp = requests.post(
            SNOW_UPDATE_URL,
            headers={
                "X-API-KEY":    SNOW_UPDATE_KEY,
                "Content-Type": "application/json",
                "Accept":       "application/json"
            },
            json={
                "RequestData": {
                    "Source": "Code Warehouse Automation",
                    "IncidentDetails": {
                        "IncidentNumber":     incident_number,
                        "AdditionalComments": clean_report
                    }
                }
            },
            timeout=30,
            verify=False
        )
        update_resp.raise_for_status()
        pipeline["steps"]["servicenow_update"] = "success"
        pipeline["snow_update_response"]       = update_resp.json()
    except Exception as e:
        pipeline["steps"]["servicenow_update"] = f"FAILED: {e}"

    # ── 7. Return ─────────────────────────────────────────────────────────
    pipeline["final_report"]  = final_report
    pipeline["top_3_similar"] = top_3
    pipeline["tidal_raw"]     = tidal_output
    pipeline["tidal_length"]  = len(tidal_output)

    return func.HttpResponse(
        json.dumps(pipeline, indent=2),
        status_code=200,
        mimetype="application/json"
    )
