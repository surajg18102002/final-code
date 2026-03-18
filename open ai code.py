"""
OpenAI/GPT Helper Module
Handles all Azure OpenAI GPT operations for incident analysis and report generation.
Switched from azure-ai-inference (model-router) to openai SDK with gpt-5.2 endpoint.
"""

import logging
import os
from openai import AzureOpenAI


# GPT Configuration
GPT_ENDPOINT    = os.environ.get("GPT_ENDPOINT", "")
GPT_KEY         = os.environ.get("GPT_KEY", "")
GPT_MODEL       = os.environ.get("GPT_MODEL", "gpt-5.2")
GPT_API_VERSION = os.environ.get("GPT_API_VERSION", "2025-04-01-preview")


def _get_client() -> AzureOpenAI:
    """
    Create and return an AzureOpenAI client using environment config.

    Returns:
        AzureOpenAI: Configured client instance
    """
    return AzureOpenAI(
        azure_endpoint=GPT_ENDPOINT,
        api_key=GPT_KEY,
        api_version=GPT_API_VERSION
    )


def generate_incident_analysis(incident_number: str, short_desc: str, assignment_group: str,
                               config_item: str, job_name: str, job_id: str,
                               tidal_output: str, similar_incidents: list) -> str:
    """
    Generate GPT-powered incident analysis report for Tidal (CSDT CSDT) incidents.

    Args:
        incident_number: ServiceNow incident number
        short_desc: Short description of the incident
        assignment_group: Assignment group from ServiceNow
        config_item: Configuration item
        job_name: Name of the Tidal job
        job_id: Tidal job run ID
        tidal_output: Output from Tidal job
        similar_incidents: List of similar incidents from knowledge base

    Returns:
        str: GPT-generated analysis report

    Raises:
        Exception: If GPT API call fails
    """
    try:
        logging.info(f"Generating GPT analysis for incident {incident_number}")
        logging.info(f"GPT Config - Endpoint: {GPT_ENDPOINT[:50]}... Model: {GPT_MODEL}")

        # Validate configuration
        if not GPT_ENDPOINT:
            raise ValueError("GPT_ENDPOINT is not configured")
        if not GPT_KEY:
            raise ValueError("GPT_KEY is not configured")
        if not GPT_MODEL:
            raise ValueError("GPT_MODEL is not configured")

        # Format similar incidents for the prompt
        similar_incidents_text = _format_similar_incidents(similar_incidents)

        # Define system and user prompts
        system_prompt = _get_system_prompt()
        user_prompt = _get_user_prompt(
            incident_number, config_item, job_name, job_id,
            tidal_output, similar_incidents_text
        )

        # Call GPT API
        logging.info(f"Calling GPT API with model: {GPT_MODEL}")
        logging.info(f"System prompt length: {len(system_prompt)} chars (~{len(system_prompt)//4} tokens)")
        logging.info(f"User prompt length: {len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")
        logging.info(f"Total prompt estimate: ~{(len(system_prompt) + len(user_prompt))//4} tokens")

        gpt_client = _get_client()
        gpt_response = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2
        )

        # Log response details
        logging.info(f"GPT response type: {type(gpt_response)}")
        logging.info(f"GPT response has choices: {hasattr(gpt_response, 'choices')}")

        # Extract response content
        if not gpt_response:
            raise ValueError("GPT API returned None response")

        if not hasattr(gpt_response, 'choices') or not gpt_response.choices:
            logging.error(f"GPT response object: {gpt_response}")
            raise ValueError("GPT API returned response without choices")

        logging.info(f"Number of choices: {len(gpt_response.choices)}")

        # Log finish_reason for debugging
        first_choice = gpt_response.choices[0]
        finish_reason = first_choice.finish_reason if hasattr(first_choice, 'finish_reason') else 'unknown'
        logging.info(f"Finish reason: {finish_reason}")

        if not first_choice.message:
            raise ValueError("GPT API choice has no message")

        message = first_choice.message
        logging.info(f"GPT message object: {message}")

        final_report = message.content

        if not final_report:
            logging.error(f"GPT message refusal: {message.refusal if hasattr(message, 'refusal') else 'N/A'}")
            logging.error(f"Finish reason: {finish_reason}")

            if finish_reason == 'length':
                raise ValueError(f"GPT response hit token limit. finish_reason: {finish_reason}")

            logging.error(f"GPT Endpoint configured: {bool(GPT_ENDPOINT)}")
            logging.error(f"GPT Key configured: {bool(GPT_KEY)}")
            logging.error(f"GPT Model: {GPT_MODEL}")
            raise ValueError("GPT API returned empty content in message")

        if not final_report.strip():
            raise ValueError(f"GPT API returned whitespace-only content: '{final_report}'")

        # Remove leading separator lines (any line containing only '=' characters)
        final_report = final_report.strip()
        while final_report:
            first_line = final_report.split('\n', 1)[0].strip()
            if first_line and all(c == '=' for c in first_line):
                lines = final_report.split('\n', 1)
                final_report = lines[1] if len(lines) > 1 else ''
                final_report = final_report.strip()
            else:
                break

        # Remove trailing separator lines
        while final_report:
            last_line = final_report.rsplit('\n', 1)[-1].strip()
            if last_line and all(c == '=' for c in last_line):
                lines = final_report.rsplit('\n', 1)
                final_report = lines[0] if len(lines) > 1 else ''
                final_report = final_report.strip()
            else:
                break

        logging.info(f"Successfully generated GPT analysis: {len(final_report)} chars")
        return final_report

    except ValueError as e:
        logging.error(f"GPT validation error: {e}")
        logging.error(f"GPT Endpoint configured: {bool(GPT_ENDPOINT)}")
        logging.error(f"GPT Key configured: {bool(GPT_KEY)}")
        logging.error(f"GPT Model: {GPT_MODEL}")
        raise
    except Exception as e:
        logging.error(f"GPT analysis failed with {type(e).__name__}: {e}")
        logging.error(f"GPT Endpoint: {GPT_ENDPOINT[:100] if GPT_ENDPOINT else 'NOT SET'}")
        logging.error(f"GPT Model: {GPT_MODEL}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


def generate_general_incident_analysis(incident_number: str, short_desc: str, assignment_group: str,
                                       config_item: str, work_notes: str,
                                       similar_incidents: list) -> str:
    """
    Generate GPT-powered incident analysis report for non-Tidal (general) incidents.
    Uses ShortDescription + WorkNotes as context instead of a Tidal error log.

    Args:
        incident_number: ServiceNow incident number
        short_desc: Short description of the incident
        assignment_group: Assignment group from ServiceNow
        config_item: Configuration item from ServiceNow
        work_notes: Work notes from the current incident
        similar_incidents: List of similar incidents from knowledge base

    Returns:
        str: GPT-generated analysis report

    Raises:
        Exception: If GPT API call fails
    """
    try:
        logging.info(f"Generating general GPT analysis for incident {incident_number}")

        if not GPT_ENDPOINT:
            raise ValueError("GPT_ENDPOINT is not configured")
        if not GPT_KEY:
            raise ValueError("GPT_KEY is not configured")
        if not GPT_MODEL:
            raise ValueError("GPT_MODEL is not configured")

        similar_incidents_text = _format_similar_incidents(similar_incidents)

        system_prompt = (
            "You are an IT Operations analyst. Analyze IT incidents and create clear, structured ServiceNow work notes. "
            "Follow the exact format provided. Be specific and actionable. Use plain text only - no markdown bold (**), just plain text."
        )

        user_prompt = _get_general_user_prompt(
            incident_number, short_desc, assignment_group, config_item,
            work_notes, similar_incidents_text
        )

        logging.info(f"Calling GPT API (general) with model: {GPT_MODEL}")

        gpt_client = _get_client()
        gpt_response = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2
        )

        if not gpt_response or not hasattr(gpt_response, 'choices') or not gpt_response.choices:
            raise ValueError("GPT API returned no choices")

        first_choice = gpt_response.choices[0]
        finish_reason = getattr(first_choice, 'finish_reason', 'unknown')
        logging.info(f"General GPT finish reason: {finish_reason}")

        if not first_choice.message:
            raise ValueError("GPT API choice has no message")

        final_report = first_choice.message.content

        if not final_report or not final_report.strip():
            if finish_reason == 'length':
                raise ValueError(f"GPT hit token limit. finish_reason: {finish_reason}")
            raise ValueError("GPT API returned empty content")

        # Strip leading/trailing separator lines
        final_report = final_report.strip()
        while final_report:
            first_line = final_report.split('\n', 1)[0].strip()
            if first_line and all(c == '=' for c in first_line):
                lines = final_report.split('\n', 1)
                final_report = (lines[1] if len(lines) > 1 else '').strip()
            else:
                break
        while final_report:
            last_line = final_report.rsplit('\n', 1)[-1].strip()
            if last_line and all(c == '=' for c in last_line):
                lines = final_report.rsplit('\n', 1)
                final_report = (lines[0] if len(lines) > 1 else '').strip()
            else:
                break

        logging.info(f"Successfully generated general GPT analysis: {len(final_report)} chars")
        return final_report

    except Exception as e:
        logging.error(f"General GPT analysis failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def generate_incident_summary(ticket_number: str, config_item: str, short_description: str,
                               work_notes: str, resolution_notes: str) -> str:
    """
    Generate AI-based summary of work notes and resolution for a similar incident.

    Args:
        ticket_number: Incident ticket number
        config_item: Configuration item/job name
        short_description: Short description of the incident
        work_notes: Work notes from the incident
        resolution_notes: Resolution notes from the incident

    Returns:
        str: AI-generated summary of the incident resolution
    """
    try:
        # Validate configuration
        if not GPT_ENDPOINT or not GPT_KEY or not GPT_MODEL:
            logging.warning("GPT not configured for summary generation")
            return "AI summary unavailable (GPT not configured)."

        # Combine notes for analysis
        combined_notes = f"{resolution_notes}\n\n{work_notes}"
        combined_notes = combined_notes[:1500]  # Limit length

        # Skip if no meaningful content
        if not combined_notes or len(combined_notes.strip()) < 30:
            logging.info(f"Skipping summary for {ticket_number} - insufficient content")
            return "No resolution details available in this ticket."

        system_prompt = (
            "You are a technical support analyst. Create brief incident resolution summaries."
        )

        user_prompt = (
            f"Create a brief 2-sentence summary of how this IT incident was resolved:\n\n"
            f"Ticket {ticket_number} for job {config_item}:\n"
            f"{combined_notes}\n\n"
            f"Summary:"
        )

        logging.info(f"Generating summary for {ticket_number}...")

        gpt_client = _get_client()
        response = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7
        )

        logging.info(f"Response received for {ticket_number}, checking content...")

        if not response:
            logging.warning(f"No response object for {ticket_number}")
            return "No response from AI."

        if not hasattr(response, 'choices') or not response.choices:
            logging.warning(f"No choices in response for {ticket_number}")
            return "No response choices from AI."

        first_choice = response.choices[0]
        logging.info(f"First choice finish_reason: {getattr(first_choice, 'finish_reason', 'unknown')}")

        if not hasattr(first_choice, 'message') or not first_choice.message:
            logging.warning(f"No message in first choice for {ticket_number}")
            return "No message in AI response."

        message = first_choice.message
        summary = getattr(message, 'content', None)

        if summary and summary.strip():
            logging.info(f"Summary generated for {ticket_number}: {summary[:50]}...")
            return summary.strip()
        else:
            logging.warning(f"Empty summary content for {ticket_number}. Message: {message}")

            if hasattr(message, 'refusal') and message.refusal:
                logging.warning(f"AI refused to generate summary: {message.refusal}")
                return f"AI declined: {message.refusal[:50]}"

            # Fallback: use resolution or work notes directly
            if resolution_notes and len(resolution_notes.strip()) > 30:
                fallback = resolution_notes.strip()[:200]
                logging.info(f"Using fallback summary from resolution notes for {ticket_number}")
                return f"{fallback}..."
            elif work_notes and len(work_notes.strip()) > 30:
                fallback = work_notes.strip()[:200]
                logging.info(f"Using fallback summary from work notes for {ticket_number}")
                return f"{fallback}..."

            return "Resolution details available in ticket history."

    except Exception as e:
        logging.warning(f"Failed to generate summary for {ticket_number}: {e}")
        return f"Summary generation failed: {str(e)[:50]}"


def test_gpt_connection() -> bool:
    """
    Test GPT API connection and configuration.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        gpt_client = _get_client()
        response = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": "Say 'test successful' if you can read this."}
            ],
            temperature=0
        )
        result = response.choices[0].message.content
        logging.info(f"GPT connection test result: {result}")
        return True

    except Exception as e:
        logging.error(f"GPT connection test failed: {e}")
        return False


def _format_similar_incidents(similar_incidents: list) -> str:
    """
    Format similar incidents list for GPT prompt with detailed multi-line format.

    Args:
        similar_incidents: List of similar incident dictionaries

    Returns:
        str: Formatted text of similar incidents with indented details
    """
    if not similar_incidents:
        return "No similar incidents found in knowledge base."

    formatted_text = ""
    for i, ticket in enumerate(similar_incidents, 1):
        ticket_num  = ticket['ticket_number']
        config      = ticket.get('config_item', 'N/A')
        short_desc  = ticket.get('short_description', 'N/A')
        resolution  = ticket.get('resolution_details', ticket.get('resolution_action', 'N/A'))

        formatted_text += f"   Ticket: {ticket_num}\n"
        formatted_text += f"   Config Item: {config}\n"
        formatted_text += f"   Short Description: {short_desc}\n"
        formatted_text += f"   Resolution: {resolution}\n"

        if ticket.get('ai_summary'):
            formatted_text += f"   Work Notes & Resolution Summary: {ticket['ai_summary']}\n"

        if i < len(similar_incidents):
            formatted_text += "\n"

    return formatted_text


def _get_system_prompt() -> str:
    """
    Get the system prompt for Tidal GPT analysis.

    Returns:
        str: System prompt defining GPT's role and behavior
    """
    return (
        "You are an IT Operations analyst. Analyze Tidal job failures and create clear, structured ServiceNow work notes. "
        "Follow the exact format provided. Be specific and actionable. Use plain text only - no markdown, bullets, or special characters."
    )


def _get_user_prompt(incident_number: str, config_item: str, job_name: str, job_id: str,
                     tidal_output: str, similar_incidents_text: str) -> str:
    """
    Construct the user prompt for Tidal incident GPT analysis.

    Args:
        incident_number: ServiceNow incident number
        config_item: Configuration item
        job_name: Tidal job name
        job_id: Tidal job run ID
        tidal_output: Tidal job output (full error log)
        similar_incidents_text: Formatted similar incidents text

    Returns:
        str: Complete user prompt for GPT
    """
    tidal_log_text = tidal_output if tidal_output else 'Error log unavailable'

    return (
        f"Analyze this Tidal job failure and create a ServiceNow work note.\n\n"

        f"INPUT DATA:\n"
        f"Incident: {incident_number}\n"
        f"Job: {job_name} (Run ID: {job_id})\n"
        f"Config Item: {config_item}\n\n"

        f"TIDAL ERROR LOG:\n"
        f"{tidal_log_text}\n\n"

        f"SIMILAR PAST INCIDENTS:\n"
        f"{similar_incidents_text}\n\n"

        f"CREATE WORK NOTE WITH EXACTLY THIS FORMAT:\n\n"

        f"⚠️ 1. JOB FAILURE SUMMARY\n"
        f"• {incident_number}: [Write 2-3 sentences summarizing what failed, including job name, run ID, and error description.]\n\n"

        f"❌ 2. ERROR MESSAGE\n"
        f">> Exact error from Tidal log of {job_name} (Job run {job_id}):\n"
        f"[Paste the COMPLETE error message from the Tidal log above. Include all error details, stack traces, XML content, and Java exceptions.]\n\n"

        f"📋 3. RELATED PAST INCIDENTS\n"
        f"[Copy each similar incident EXACTLY as formatted with indentation:\n"
        f"   Ticket: [number]\n"
        f"   Config Item: [name]\n"
        f"   Short Description: [description]\n"
        f"   Resolution: [details]\n"
        f"   Work Notes & Resolution Summary: [AI summary if available]\n"
        f"]\n\n"

        f"🔧 4. AI RECOMMENDED RESOLUTION STEPS\n"
        f"[Provide 5-7 numbered steps (1., 2., 3., etc.) with specific technical actions referencing systems, files, and agents from the error log.]\n\n"

        f"REQUIREMENTS:\n"
        f"- Use bullet points (•) for lists in sections 1 and 3\n"
        f"- Use numbered lists (1. 2. 3.) for section 4\n"
        f"- Keep all separator lines (============)\n"
        f"- NO markdown bold (**) - use plain text only\n"
        f"- Keep emojis (⚠️ ❌ 📋 🔧) at start of section titles\n"
        f"- Total length 400-600 words"
    )


def _get_general_user_prompt(incident_number: str, short_desc: str, assignment_group: str,
                              config_item: str, work_notes: str,
                              similar_incidents_text: str) -> str:
    """
    Construct the user prompt for general (non-Tidal) incident GPT analysis.

    Args:
        incident_number: ServiceNow incident number
        short_desc: Short description of the incident
        assignment_group: Assignment group
        config_item: Configuration item
        work_notes: Work notes from the current incident (used as error context)
        similar_incidents_text: Formatted similar incidents text

    Returns:
        str: Complete user prompt for GPT
    """
    incident_context = (
        f"Short Description: {short_desc}\n\nWork Notes:\n{work_notes}"
        if work_notes else f"Short Description: {short_desc}"
    )
    incident_context = incident_context[:3000]

    return (
        f"Analyze this IT incident and create a ServiceNow work note.\n\n"

        f"INPUT DATA:\n"
        f"Incident: {incident_number}\n"
        f"Configuration Item: {config_item}\n"
        f"Assignment Group: {assignment_group}\n\n"

        f"INCIDENT DETAILS:\n"
        f"{incident_context}\n\n"

        f"SIMILAR PAST INCIDENTS:\n"
        f"{similar_incidents_text}\n\n"

        f"CREATE WORK NOTE WITH EXACTLY THIS FORMAT:\n\n"

        f"⚠️ 1. INCIDENT SUMMARY\n"
        f"• {incident_number}: [Write 2-3 sentences summarizing the issue, including the configuration item and what went wrong based on the incident details.]\n\n"

        f"❌ 2. ISSUE DETAILS\n"
        f">> Key details from {config_item} incident {incident_number}:\n"
        f"[Summarize the key error or problem from the incident details above. Include relevant symptoms, error messages, or failure indicators.]\n\n"

        f"📋 3. RELATED PAST INCIDENTS\n"
        f"[Copy each similar incident EXACTLY as formatted with indentation:\n"
        f"   Ticket: [number]\n"
        f"   Config Item: [name]\n"
        f"   Short Description: [description]\n"
        f"   Resolution: [details]\n"
        f"   Work Notes & Resolution Summary: [AI summary if available]\n"
        f"]\n\n"

        f"🔧 4. AI RECOMMENDED RESOLUTION STEPS\n"
        f"[Provide 5-7 numbered steps (1., 2., 3., etc.) with specific technical actions based on the incident details and how similar incidents were resolved.]\n\n"

        f"REQUIREMENTS:\n"
        f"- Use bullet points (•) for lists in sections 1 and 3\n"
        f"- Use numbered lists (1. 2. 3.) for section 4\n"
        f"- Keep all separator lines (============)\n"
        f"- NO markdown bold (**) - use plain text only\n"
        f"- Keep emojis (⚠️ ❌ 📋 🔧) at start of section titles\n"
        f"- Total length 400-600 words"
    )
