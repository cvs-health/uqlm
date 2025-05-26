"""
Provides an agent for troubleshooting UiPath Orchestrator jobs.
"""
import os
from typing import Any, Dict, Optional
import requests # For potential HTTP errors if UiPathError is not comprehensive

try:
    from uipath import UiPathRetrySession, UiPathError
    UIPATH_SDK_INSTALLED = True
except ImportError:
    UiPathRetrySession = None # type: ignore
    UiPathError = None # type: ignore
    UIPATH_SDK_INSTALLED = False

# UQLM specific imports for LLM and UQ
from .uipath_llms import get_uipath_llm, invoke_uipath_llm
from ..scorers.black_box import BlackBoxUQ # Using existing BlackBoxUQ as a UQ scorer example

class OrchestratorTroubleshootingAgent:
    """
    An agent designed to troubleshoot UiPath Orchestrator jobs by fetching
    details, logs, and potentially running diagnostic steps.

    Authentication with UiPath Orchestrator is typically handled via environment
    variables configured by the UiPath SDK (`uipath auth` CLI or manual setup).
    Required environment variables:
    - UIPATH_URL: Your UiPath Automation Cloud URL (e.g., https://cloud.uipath.com/<org>/<tenant>).
    - UIPATH_CLIENT_ID: Your UiPath application client ID (for OAuth).
    - UIPATH_CLIENT_SECRET: Your UiPath application client secret (for OAuth).
    OR
    - UIPATH_ACCESS_TOKEN: A direct access token.
    
    Optional folder context variables:
    - UIPATH_FOLDER_PATH: Full path to the target Orchestrator folder.
    - UIPATH_FOLDER_KEY: Unique key/ID of the target Orchestrator folder.
    If `orchestrator_folder` is provided to the constructor, it may be used to
    set request headers if not automatically handled by the SDK based on these env vars.
    """

    def __init__(
        self,
        jobKey: str,
        orchestrator_folder: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initializes the OrchestratorTroubleshootingAgent.

        Args:
            jobKey: The unique key (GUID) or ID of the Orchestrator job to troubleshoot.
            orchestrator_folder: Optional. The Orchestrator folder name or ID where the job resides.
                                 If None, the SDK might use the folder context from environment
                                 variables (UIPATH_FOLDER_PATH or UIPATH_FOLDER_KEY).
            **kwargs: Additional keyword arguments. Can include:
                        - `uipath_client`: An already initialized UiPathRetrySession or compatible client.
        """
        if not UIPATH_SDK_INSTALLED:
            raise ImportError("UiPath SDK (uipath) is not installed. Please install it via `pip install uipath`.")
        if not jobKey:
            raise ValueError("jobKey cannot be empty.")

        self.job_key: str = str(jobKey) # Ensure jobKey is a string for URL construction
        self.orchestrator_folder: Optional[str] = orchestrator_folder
        self.additional_params: Dict[str, Any] = kwargs

        self.orchestrator_client = kwargs.get('uipath_client')
        if not self.orchestrator_client:
            try:
                self.orchestrator_client = UiPathRetrySession()
                # The UiPathRetrySession should automatically pick up credentials from env vars.
            except Exception as e: # Broad exception if UiPathRetrySession() itself fails
                raise RuntimeError(f"Failed to initialize UiPathRetrySession: {e}. "
                                   "Ensure UiPath SDK is configured correctly with environment variables.")

        # Store UIPATH_URL for constructing direct OData URLs if necessary.
        # Prefer SDK methods if they abstract this away.
        self.uipath_url = os.getenv("UIPATH_URL")
        if not self.uipath_url:
            print("Warning: UIPATH_URL environment variable not found. Direct OData calls might fail.")
            # Depending on UiPathRetrySession, it might still work if base_url is configured internally.

        print(f"OrchestratorTroubleshootingAgent initialized for jobKey: {self.job_key}")
        if self.orchestrator_folder:
            print(f"Targeting Orchestrator Folder: {self.orchestrator_folder} (Note: SDK may use UIPATH_FOLDER_PATH/KEY env vars by default)")


    def get_job_details(self) -> Dict[str, Any]:
        """
        Fetches detailed information about the specified Orchestrator job using OData.

        Information could include job state, start/end times, input/output arguments,
        logs, error messages, and related entities like process, robot, machine.

        Returns:
            A dictionary containing the job details.
        
        Raises:
            RuntimeError: If fetching job details fails due to API errors,
                          authentication issues, or network problems.
            ValueError: If UIPATH_URL is not configured and required.
        """
        if not self.orchestrator_client:
             raise RuntimeError("UiPath client not initialized.")
        if not self.uipath_url:
            raise ValueError("UIPATH_URL is not configured, cannot construct OData request URL.")

        # Construct the OData URL for fetching a specific job by its key.
        # The job_key in Orchestrator is typically a long integer ID, not a GUID for direct OData access by key.
        # If jobKey is a GUID, the endpoint might be different or require filtering.
        # Assuming job_key is the numeric ID for /odata/Jobs(ID).
        odata_url = f"{self.uipath_url.rstrip('/')}/odata/Jobs({self.job_key})"
        
        headers = {
            "Accept": "application/json"
            # Folder context: UiPathRetrySession might handle this based on env vars
            # (UIPATH_FOLDER_KEY or UIPATH_FOLDER_PATH).
            # If orchestrator_folder was a numeric ID, we could add:
            # "X-UIPATH-OrganizationUnitId": str(self.orchestrator_folder)
            # If it's a path, it needs to be resolved to an ID first, which is complex here.
            # For now, relying on SDK's default folder context from environment.
        }
        if self.orchestrator_folder and self.orchestrator_folder.isdigit():
             # Basic assumption: if orchestrator_folder is an ID, use it.
             # This might conflict if UIPATH_FOLDER_KEY is also set. SDK behavior is key.
             headers["X-UIPATH-OrganizationUnitId"] = self.orchestrator_folder

        print(f"Attempting to fetch details for jobKey: {self.job_key} from {odata_url}")

        try:
            response = self.orchestrator_client.get(odata_url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return response.json()  # Assuming the response is JSON
        except UiPathError as e: # Specific SDK error
            raise RuntimeError(f"UiPath API error fetching job details for job {self.job_key}: {e}")
        except requests.exceptions.HTTPError as e: # HTTP errors like 404, 500
            error_content = e.response.text if e.response else "No response content"
            raise RuntimeError(f"HTTP error fetching job details for job {self.job_key}: {e.response.status_code} - {error_content}")
        except requests.exceptions.RequestException as e: # Other network/request errors
            raise RuntimeError(f"Request error fetching job details for job {self.job_key}: {e}")
        except ValueError as e: # JSON decoding error
            raise RuntimeError(f"Failed to decode JSON response for job {self.job_key}: {e}")
        except Exception as e: # Catch-all for other unexpected errors
            raise RuntimeError(f"An unexpected error occurred while fetching job details for {self.job_key}: {e}")

    def troubleshoot(self) -> str:
        Performs troubleshooting steps for the Orchestrator job.

        This method would orchestrate calls to get_job_details(), analyze logs,
        check common failure points, and potentially use an LLM to generate
        a summary or recommendations.

        Returns:
            A string containing the troubleshooting summary or diagnosis.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        print(f"Starting troubleshooting for jobKey: {self.job_key}...")
        
        try:
            job_details = self.get_job_details()
            # TODO: Analyze job_details and logs.
            # TODO: Implement logic to identify common issues.
            # TODO: Potentially use an LLM to summarize findings or suggest solutions.
            summary = f"Troubleshooting for job {self.job_key} based on details: {job_details}."
        except NotImplementedError:
            summary = (
                f"Troubleshooting for job {self.job_key} cannot proceed "
                "because get_job_details() is not implemented."
            )
        except Exception as e:
            summary = f"An error occurred during troubleshooting for job {self.job_key}: {e}"
            # Log the exception traceback here in a real scenario

        # For now, returning a placeholder that indicates the method was called.
        # In the future, this will be a more complex analysis result.
        if "NotImplementedError" in summary:
             return summary # Return the specific NotImplementedError message for now
        
        # Placeholder until actual logic is added and LLM/UQ steps are implemented
        # For now, returning a string that includes some job details if fetched, 
        # or the error encountered during get_job_details().
        
        if isinstance(job_details, dict): # Check if job_details were successfully fetched
            # Example: Extract some basic info if available
            job_id_info = job_details.get('Id', self.job_key) # Prefer Id from details if available
            job_state_info = job_details.get('State', 'N/A')
            formatted_details = (
                f"Job Key: {job_id_info}\n"
                f"Job State: {job_state_info}\n"
                # Add more details as needed, e.g., job_details.get('Info', 'N/A')
            )
            # TODO: 1. Extract relevant information (errors, logs) from job_details
            # relevant_info = {"errors": job_details.get("OutputArguments", {}).get("Errors"), "logs": "..."}
            
            # TODO: 2. Initialize a UiPath LLM client using get_uipath_llm()
            # try:
            #     llm_model_name = self.additional_params.get("llm_model_name", "default-uipath-llm")
            #     llm_client = get_uipath_llm(llm_name=llm_model_name)
            # except Exception as llm_e:
            #     return f"Failed to initialize LLM: {llm_e}\nJob Details:\n{formatted_details}"

            # TODO: 3. Construct a prompt for the LLM to suggest troubleshooting steps
            # prompt = f"Troubleshoot Orchestrator job with details:\n{formatted_details}\nErrors: {relevant_info['errors']}"
            
            # TODO: 4. Invoke the LLM using invoke_uipath_llm() to get suggested solutions
            # try:
            #     suggested_solutions_raw = invoke_uipath_llm(llm_client, prompt)
            #     # suggested_solutions_list = parse_solutions(suggested_solutions_raw) # Assuming LLM gives multiple
            # except Exception as invoke_e:
            #     return f"Failed to invoke LLM: {invoke_e}\nJob Details:\n{formatted_details}"

            # TODO: 5. Initialize a UQ scorer (e.g., BlackBoxUQ)
            # uq_scorer = BlackBoxUQ(...) # Initialize with appropriate parameters

            # TODO: 6. Score the generated solutions using the UQ scorer (conceptual)
            # scored_solutions = []
            # for sol in suggested_solutions_list:
            #     score = uq_scorer.score(prompt, sol) # This is highly conceptual
            #     scored_solutions.append({"solution": sol, "score": score})
            
            # TODO: 7. Select the best solution based on UQ scores
            # best_solution = max(scored_solutions, key=lambda x: x['score']['score']) # Example

            # Current placeholder return
            return (
                f"Successfully fetched details for Job Key: {self.job_key}.\n"
                f"{formatted_details}\n"
                "Troubleshooting steps would be generated here using LLM and UQ."
            )
        else: # job_details is likely an error message string from the except block
            return job_details

    def send_solution_to_slack(self, solution_steps: str, slack_channel: Optional[str] = None) -> bool:
        """
        Sends the provided troubleshooting solution steps to a specified Slack channel.

        This method would typically use a UiPath Slack connector, a generic Slack API
        via UiPath HTTP activities, or a direct Slack client library.
        Configuration details such as Slack webhook URL, bot token, and default channel
        might be retrieved from environment variables or UiPath Orchestrator assets.

        Args:
            solution_steps: A string containing the troubleshooting steps/solution to send.
            slack_channel: Optional. The specific Slack channel to send the message to.
                           If None, a default channel might be used (from config).

        Returns:
            True if the message was sent successfully (placeholder), False otherwise.
        """
        target_channel = slack_channel or os.getenv("SLACK_DEFAULT_CHANNEL") or "default-general"
        print(f"Attempting to send solution to Slack channel: {target_channel} for job {self.job_key}")
        print(f"Solution:\n{solution_steps}")

        # TODO: 1. Retrieve Slack configuration (e.g., webhook URL, bot token)
        #           from environment variables or UiPath Orchestrator assets.
        # slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        # slack_bot_token = os.getenv("SLACK_BOT_TOKEN") # If using Web API instead of webhook

        # TODO: 2. Initialize UiPath SDK client for Slack (if available and applicable)
        #           or prepare for a direct API call (e.g. using requests or UiPathRetrySession).
        #           If using self.orchestrator_client for a generic HTTP request:
        #           Ensure it's configured for the Slack API base URL if different.

        # TODO: 3. Construct the payload for the Slack message.
        # payload = {
        #     "channel": target_channel, # Or not needed if webhook is channel-specific
        #     "text": f"Troubleshooting solution for Orchestrator Job ID {self.job_key}:\n{solution_steps}"
        # }
        # if slack_bot_token: headers = {"Authorization": f"Bearer {slack_bot_token}"} else: headers = {}

        # TODO: 4. Make the API call to send the message.
        # try:
        #     # Example using a webhook:
        #     # response = requests.post(slack_webhook_url, json=payload)
        #     # response.raise_for_status()
        #     # Or using UiPathRetrySession for a generic POST:
        #     # response = self.orchestrator_client.post(slack_api_endpoint, json=payload, headers=headers)
        #     # response.raise_for_status()
        #     print("Placeholder: Slack message sent successfully.")
        #     return True
        # except Exception as e:
        #     print(f"Error sending Slack message: {e}")
        #     return False
        
        print("Placeholder: Slack integration not fully implemented. Simulating success.")
        return True # Placeholder return

    def __repr__(self) -> str:
        return (
            f"OrchestratorTroubleshootingAgent(jobKey='{self.job_key}', "
            f"orchestrator_folder='{self.orchestrator_folder}')"
        )
