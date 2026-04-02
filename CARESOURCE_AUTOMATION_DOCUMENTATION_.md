# ClinAuto – Clinical Support Services Automation
## Technical Documentation: Azure Function App

**Repository:** [CareSourceIT/ClinAuto-Clinical.Support.Services.automation](https://github.com/CareSourceIT/ClinAuto-Clinical.Support.Services.automation/tree/feature/auto_az_function)
**Branch:** `feature/auto_az_function`
**Last Updated:** April 2026

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [How It Works – End-to-End Flow](#3-how-it-works--end-to-end-flow)
4. [API Endpoints](#4-api-endpoints)
5. [Helper Modules](#5-helper-modules)
6. [Environment Variables / Configuration](#6-environment-variables--configuration)
7. [Dependencies](#7-dependencies)
8. [Incident Type Routing (Tidal vs General)](#8-incident-type-routing-tidal-vs-general)
9. [Similarity Search Algorithm](#9-similarity-search-algorithm)
10. [ServiceNow Integration](#10-servicenow-integration)

---

## 1. Architecture Overview

This is a **Python Azure Function App** (v2 programming model) triggered via HTTP. It is invoked by an **Azure Logic App** which monitors an email inbox for incoming incident notifications. When an email arrives with a ServiceNow incident number in the subject, the Logic App calls the Azure Function with the incident number.

```
Email Mailbox
      │  (Logic App triggered on email arrival)
      ▼
Azure Logic App
      │  HTTP POST → { "incident_number": "INC1234567" }
      ▼
Azure Function App (Python)
      │
      ├── GET  /api/health                  → Health check (anonymous)
      ├── POST /api/find-similar-tickets     → Analysis only, no SNOW update
      └── POST /api/process-incident         → Full analysis + updates ServiceNow
```

---

## 2. Repository Structure

```
feature/auto_az_function/
├── function_app.py                  # Main entry point — all routes defined here
├── host.json                        # Azure Function runtime configuration
├── requirements.txt                 # Python dependencies
└── helpers/
    ├── servicenow_helper.py         # ServiceNow API fetch + update operations
    ├── openai_helper.py             # Azure OpenAI GPT analysis functions
    ├── tidal_helper.py              # Tidal job scheduler API integration
    └── blob_storage_helper.py       # Azure Blob Storage knowledge base + similarity search
```

---

## 3. How It Works – End-to-End Flow

### Shared Analysis Pipeline (`_run_analysis_pipeline`)

Both main routes call a shared internal function that runs these 4 steps in order:

```
Step 1: Fetch incident details from ServiceNow
           ↓
        Parse fields:
          - ShortDescription, ConfigurationItem, WorkNotes
          - AssignmentGroup, RequestedFor, job_name, job_id
           ↓
Step 2: Route by incident type
           │
           ├── RequestedFor == "CSDT CSDT"  →  TIDAL INCIDENT
           │      └─ Fetch job log output from Tidal API using job_id
           │         (XML payload → parsed job output text)
           │
           └── Anything else  →  GENERAL INCIDENT
                  └─ Use ShortDescription + WorkNotes as context
           ↓
Step 3: Similarity search (Azure Blob Storage knowledge base)
           - Load historical incidents CSV from blob storage
           - TF-IDF vectorization with cosine similarity
           - Strict exact config_item/job_name matching only
           - Returns top 3 most similar historical tickets
           ↓
Step 4: Generate AI summaries for each similar ticket (2-sentence summaries)
```

### Route 1: `/api/find-similar-tickets`
- Runs the shared pipeline above
- Calls GPT with **plain-text** prompt (no HTML)
- Returns the full pipeline result as JSON
- **Does NOT update ServiceNow**

### Route 2: `/api/process-incident`
- Runs the shared pipeline above
- Calls GPT with **HTML-formatted** prompt (for ServiceNow work notes display)
- Sanitizes the HTML output
- Wraps it in a styled header banner with timestamp
- **Posts to ServiceNow** as `AdditionalComments` using `[code]...[/code]` wrapper so SNOW renders HTML

---

## 4. API Endpoints

| Method | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | Anonymous | Health check; returns `{"status": "healthy"}` |
| `GET/POST` | `/api/find-similar-tickets` | Function key | Analysis pipeline, plain-text GPT, no SNOW update |
| `GET/POST` | `/api/process-incident` | Function key | Full pipeline, HTML GPT + ServiceNow work note update |

### Request Format
Both analysis routes accept the incident number as:
- Query parameter: `?incident_number=INC1234567`
- JSON body: `{"incident_number": "INC1234567"}`

The `INC` prefix is automatically added if missing.

### Response Format
Both routes return JSON with the full pipeline state:
```json
{
  "incident_number": "INC1234567",
  "incident_type": "tidal",
  "steps": {
    "servicenow_fetch": "success",
    "tidal_fetch": "success",
    "similarity_search": "success",
    "ai_summaries": "success",
    "gpt_analysis": "success",
    "servicenow_update": "success"
  },
  "parsed": { ... },
  "tidal_raw": "...",
  "tidal_length": 1500,
  "top_3_similar": [ ... ],
  "final_report": "...",
  "final_report_length": 3200,
  "snow_update_response": { ... }
}
```

---

## 5. Helper Modules

### `helpers/servicenow_helper.py`
Handles all ServiceNow operations.

| Function | Description |
|---|---|
| `fetch_incident_details(incident_number)` | GET request to SNOW API; returns full incident JSON |
| `parse_incident_details(snow_data)` | Extracts `short_desc`, `config_item`, `work_notes`, `assignment_group`, `requested_for`, `job_name`, `job_id` |
| `convert_to_html_format(text)` | Cleans GPT HTML output; strips stray markdown fences, `<script>` tags, and preamble text |
| `clean_for_snow(text)` | Normalizes line endings, collapses excessive blank lines, truncates to 30,000 chars |
| `update_incident_work_notes(incident_number, work_note)` | POSTs work note to SNOW via `AdditionalComments`; adds styled HTML header with timestamp; wraps in `[code]...[/code]` |

**SNOW API Endpoints used:**
- `GET https://apigateway-crt.caresource.corp/ServiceNow/Incident/1.0`
- `POST https://apigateway-crt.caresource.corp/ServiceNow/Incident/CreateUpdate/Sync/`

**Note on job name parsing:** The `ShortDescription` field uses the format `PRD Tidal Job <JobName> | <JobID> | Completed Abnormally`. The parser strips the `PRD Tidal Job` prefix and extracts only numeric digits from the job ID.

---

### `helpers/openai_helper.py`
Handles all Azure OpenAI GPT calls using the `openai` Python SDK.

| Function | Output Format | Used by Route |
|---|---|---|
| `generate_incident_analysis(...)` | **HTML** | `/process-incident` (Tidal) |
| `generate_general_incident_analysis(...)` | **HTML** | `/process-incident` (General) |
| `generate_incident_analysis_plain_text(...)` | **Plain text** | `/find-similar-tickets` (Tidal) |
| `generate_general_incident_analysis_plain_text(...)` | **Plain text** | `/find-similar-tickets` (General) |
| `generate_incident_summary(...)` | **Plain text** | Both (per similar ticket) |
| `test_gpt_connection()` | `bool` | Diagnostic use |

**GPT configuration:**
- Model: `gpt-5.2` (configurable via `GPT_MODEL` env var)
- API version: `2025-04-01-preview`
- Temperature: `0.2` for analysis, `0.7` for summaries
- Client: `AzureOpenAI` SDK

---

### `helpers/tidal_helper.py`
Handles Tidal Enterprise Scheduler API calls for fetching job run output logs.

| Function | Description |
|---|---|
| `get_tidal_headers()` | Builds Basic Auth header from username/password + API key |
| `fetch_job_output(job_id)` | POSTs XML payload to Tidal API; returns `(output_text, debug_info)` |
| `_parse_tidal_response(raw_response, job_id)` | Parses XML response; tries `root.text`, then `xs:string` namespace, then any `string` element |

**Tidal API:**
- Endpoint: `https://crt-apim.caresource.corp/TidalAutomation/`
- Protocol: HTTP POST with XML body (`JobOutput.getJobOutputRaw`)
- Auth: Basic Auth (Base64 `username:password`) + `x-api-key` header

---

### `helpers/blob_storage_helper.py`
Manages the historical incident knowledge base stored as a CSV in Azure Blob Storage.

| Function | Description |
|---|---|
| `load_knowledge_base()` | Downloads CSV from blob; normalizes columns; builds weighted `search_text`; caches in memory |
| `build_search_text(...)` | Builds weighted text: `short_description × 3`, `description × 2`, `config_item × 2`, notes `× 1` |
| `expand_query(query_text)` | Adds synonym expansions (e.g. `fail` → `failed, failure, error, abort, ...`) |
| `extract_resolution_action(...)` | Extracts meaningful resolution text; strips boilerplate; max 500 chars |
| `find_similar_tickets(df, config_item, query_text, top_n=3)` | TF-IDF + cosine similarity with strict exact `config_item` matching |

**Knowledge base CSV columns expected:**
- `Number` → `ticket_number`
- `Configuration item` → `config_item`
- `Short description` → `short_description`
- `Description` → `description`
- `Work notes` → `work_notes`
- `Resolution notes` → `resolution_notes`

---

## 6. Environment Variables / Configuration

All secrets and connection strings are stored as **Azure Function App Application Settings** (environment variables). None are hardcoded.

| Variable | Used in | Description |
|---|---|---|
| `SNOW_GET_KEY` | `servicenow_helper.py` | API key for reading ServiceNow incidents |
| `SNOW_UPDATE_KEY` | `servicenow_helper.py` | API key for updating ServiceNow incidents |
| `GPT_ENDPOINT` | `openai_helper.py` | Azure OpenAI endpoint URL |
| `GPT_KEY` | `openai_helper.py` | Azure OpenAI API key |
| `GPT_MODEL` | `openai_helper.py` | GPT model name (default: `gpt-5.2`) |
| `GPT_API_VERSION` | `openai_helper.py` | API version (default: `2025-04-01-preview`) |
| `TIDAL_BASE_URL` | `tidal_helper.py` | Tidal API base URL |
| `TIDAL_USERNAME` | `tidal_helper.py` | Tidal API username |
| `TIDAL_PASSWORD` | `tidal_helper.py` | Tidal API password |
| `TIDAL_API_KEY` | `tidal_helper.py` | Tidal x-api-key header value |
| `BLOB_CONNECTION_STRING` | `blob_storage_helper.py` | Azure Storage connection string |
| `BLOB_CONTAINER_NAME` | `blob_storage_helper.py` | Blob container name |
| `BLOB_FILE_NAME` | `blob_storage_helper.py` | Knowledge base CSV file name |

> ⚠️ **Important:** If any of these credentials expire, the corresponding integration will fail silently or throw an auth error. Update expired credentials via **Azure Portal → Function App → Configuration → Application Settings**.

---

## 7. Dependencies

From `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `azure-functions` | `1.20.0` | Azure Functions v2 Python SDK |
| `azure-storage-blob` | `12.28.0` | Read knowledge base CSV from blob storage |
| `openai` | `>=1.0.0` | Azure OpenAI GPT client |
| `pandas` | `>=2.0.0,<4.0.0` | Knowledge base DataFrame operations |
| `scikit-learn` | `>=1.3.0,<2.0.0` | TF-IDF vectorizer + cosine similarity |
| `openpyxl` | `3.1.2` | Excel file support (optional CSV fallback) |
| `requests` | `>=2.31.0` | HTTP calls to ServiceNow and Tidal APIs |
| `httpx` | `>=0.25.0` | Async HTTP (used by openai SDK internally) |

---

## 8. Incident Type Routing (Tidal vs General)

The function branches based on the `RequestedFor` field from ServiceNow:

```python
TIDAL_REQUESTED_FOR = "CSDT CSDT"

is_tidal = requested_for.upper() == TIDAL_REQUESTED_FOR.upper()
```

| Incident Type | `RequestedFor` value | Data Source | GPT Context |
|---|---|---|---|
| **Tidal** | `"CSDT CSDT"` | Tidal job log output (fetched via job_id) | job_name, job_id, tidal_output, similar tickets |
| **General** | Anything else | ServiceNow WorkNotes | short_desc, config_item, work_notes, similar tickets |

For **Tidal** similarity search:
- `config_item` = `job_name` (exact job name match)
- `query_text` = `{job_name} {config_item} Completed Abnormally {short_desc} {error_snippet}`

For **General** similarity search:
- `config_item` = `short_desc` (short description used as identifier)
- `query_text` = `{short_desc} {config_item} {work_notes[:500]}`

---

## 9. Similarity Search Algorithm

1. **Exact match filter:** Only tickets with `config_item` exactly matching the incoming job name are considered. If no exact matches exist, an empty list is returned (no fuzzy cross-job matches).
2. **Query expansion:** Synonyms are added to the query (e.g. `fail` → adds `failed, failure, error, abort, terminated`).
3. **TF-IDF vectorization:** `TfidfVectorizer` with `ngram_range=(1,3)`, `max_features=10000`, `sublinear_tf=True`.
4. **Cosine similarity:** Computed between the expanded query and all matching corpus documents.
5. **Relevance scoring:**
   - Base: cosine similarity score
   - +50% boost for exact `config_item` match
   - +10% per error keyword match in `short_description` (max +30%)
   - +5% per error keyword match in general text (max +15%)
6. **Quality filter:** Results below 30% of the top score are dropped.
7. Returns up to **top 3** results.

---

## 10. ServiceNow Integration

### Reading incidents
```
GET https://apigateway-crt.caresource.corp/ServiceNow/Incident/1.0
Params: IncidentNumber=INC1234567, Source=Streamline
Header: X-API-KEY: {SNOW_GET_KEY}
```

### Writing work notes
```
POST https://apigateway-crt.caresource.corp/ServiceNow/Incident/CreateUpdate/Sync/
Header: X-API-KEY: {SNOW_UPDATE_KEY}
Body:
{
  "RequestData": {
    "Source": "Code Warehouse Automation",
    "IncidentDetails": {
      "IncidentNumber": "INC1234567",
      "AdditionalComments": "[code]<styled HTML>[/code]"
    }
  }
}
```

The HTML work note is wrapped in `[code]...[/code]` so ServiceNow renders it as formatted HTML rather than raw text.

**Known behavior:** ServiceNow may return error code `21002` with a `null` message body even when the update succeeds. The code handles this by treating null-message errors as successful updates.

---

## Runtime Configuration (`host.json`)

| Setting | Value | Notes |
|---|---|---|
| Function timeout | `10 minutes` | Long enough for GPT + Tidal API calls |
| Max concurrent requests | `100` | |
| Max outstanding requests | `200` | |
| Route prefix | `api` | All routes are under `/api/...` |
| Health monitor | Enabled | 10s interval, 2min window |
| Log level | `Information` | Application Insights sampling enabled |
