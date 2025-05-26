"""Tests for the uqlm.orchestrator_agent module."""

import pytest
import os
from unittest.mock import patch, MagicMock, call

from uqlm import OrchestratorTroubleshootingAgent
import requests # For requests.exceptions

# Attempt to import UiPath SDK components for type hinting and specific mocking
try:
    from uipath import UiPathRetrySession, UiPathError
    UIPATH_SDK_AVAILABLE_FOR_TEST = True
except ImportError:
    UiPathRetrySession = MagicMock() # Mock if not available
    UiPathError = type('UiPathError', (Exception,), {}) # Dummy exception class
    UIPATH_SDK_AVAILABLE_FOR_TEST = False


@pytest.fixture
def mock_env_vars():
    """Fixture to set up necessary environment variables."""
    env_vars = {
        "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
        # Add other required auth env vars if UiPathRetrySession mock doesn't bypass them
        "UIPATH_CLIENT_ID": "dummy_client_id",
        "UIPATH_CLIENT_SECRET": "dummy_client_secret",
        # "UIPATH_ACCESS_TOKEN": "dummy_access_token" # Alternatively
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars

@pytest.fixture
def mock_uipath_retry_session():
    """Fixture to mock UiPathRetrySession."""
    if UIPATH_SDK_AVAILABLE_FOR_TEST:
        with patch('uipath.UiPathRetrySession', autospec=True) as mock_session_class:
            mock_instance = mock_session_class.return_value
            yield mock_instance
    else: # SDK not installed, patch the MagicMock we defined earlier or the import path
        with patch('uqlm.orchestrator_agent.UiPathRetrySession', autospec=True) as mock_session_class:
            mock_instance = mock_session_class.return_value
            yield mock_instance


class TestOrchestratorTroubleshootingAgentConstructor:
    """Tests for the OrchestratorTroubleshootingAgent constructor."""

    def test_init_stores_parameters_correctly(self, mock_env_vars, mock_uipath_retry_session):
        agent = OrchestratorTroubleshootingAgent(
            jobKey="12345",
            orchestrator_folder="Default",
            custom_param="test_value"
        )
        assert agent.job_key == "12345"
        assert agent.orchestrator_folder == "Default"
        assert agent.additional_params["custom_param"] == "test_value"
        assert agent.orchestrator_client is not None # Should be initialized by default

    def test_init_raises_valueerror_for_empty_jobkey(self, mock_env_vars):
        with pytest.raises(ValueError, match="jobKey cannot be empty"):
            OrchestratorTroubleshootingAgent(jobKey="")

    @patch.dict(os.environ, {"UIPATH_URL": "http://test.url"}, clear=True)
    @patch('uqlm.orchestrator_agent.UiPathRetrySession') # Patch where it's used
    def test_init_initializes_client_if_not_provided(self, mock_session_class, mock_env_vars_dict_only):
        # mock_env_vars_dict_only is not used directly but ensures os.environ is patched
        mock_instance = mock_session_class.return_value
        agent = OrchestratorTroubleshootingAgent(jobKey="123")
        mock_session_class.assert_called_once()
        assert agent.orchestrator_client == mock_instance

    def test_init_uses_provided_client(self, mock_env_vars):
        custom_client = MagicMock(spec=UiPathRetrySession if UIPATH_SDK_AVAILABLE_FOR_TEST else MagicMock)
        agent = OrchestratorTroubleshootingAgent(jobKey="123", uipath_client=custom_client)
        assert agent.orchestrator_client == custom_client
    
    @patch.dict(os.environ, {}, clear=True) # No UIPATH_URL
    def test_init_warns_if_uipath_url_missing(self, capsys, mock_uipath_retry_session):
        OrchestratorTroubleshootingAgent(jobKey="123")
        captured = capsys.readouterr()
        assert "Warning: UIPATH_URL environment variable not found" in captured.out
    
    @patch('uqlm.orchestrator_agent.UIPATH_SDK_INSTALLED', False)
    def test_init_raises_importerror_if_sdk_not_installed(self):
        with pytest.raises(ImportError, match="UiPath SDK (uipath) is not installed"):
            OrchestratorTroubleshootingAgent(jobKey="123")


class TestGetJobDetails:
    """Tests for the get_job_details method."""

    @pytest.fixture
    def agent(self, mock_env_vars, mock_uipath_retry_session):
        # mock_uipath_retry_session is used to patch the UiPathRetrySession constructor
        # The actual instance used by the agent will be this mock_instance
        agent_instance = OrchestratorTroubleshootingAgent(jobKey="67890", orchestrator_folder="TestFolder")
        # Replace the client instance on the agent with the one from the fixture if different
        agent_instance.orchestrator_client = mock_uipath_retry_session 
        return agent_instance

    def test_get_job_details_constructs_correct_url_and_calls_client(self, agent):
        mock_response = MagicMock()
        mock_response.json.return_value = {"Id": 67890, "State": "Successful"}
        agent.orchestrator_client.get.return_value = mock_response
        
        details = agent.get_job_details()

        expected_url = f"{os.getenv('UIPATH_URL')}/odata/Jobs(67890)"
        agent.orchestrator_client.get.assert_called_once_with(expected_url, headers=pytest.ANY)
        assert details == {"Id": 67890, "State": "Successful"}

    def test_get_job_details_includes_folderid_header_if_digit(self, agent):
        agent.orchestrator_folder = "12345" # Folder ID
        mock_response = MagicMock()
        mock_response.json.return_value = {"Id": 67890, "State": "Successful"}
        agent.orchestrator_client.get.return_value = mock_response

        agent.get_job_details()
        
        expected_headers = {"Accept": "application/json", "X-UIPATH-OrganizationUnitId": "12345"}
        args, kwargs = agent.orchestrator_client.get.call_args
        assert kwargs['headers'] == expected_headers
        
    def test_get_job_details_handles_httperror_404(self, agent):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404, text="Not Found"))
        agent.orchestrator_client.get.side_effect = http_error
        
        with pytest.raises(RuntimeError, match="HTTP error fetching job details.*404 - Not Found"):
            agent.get_job_details()

    def test_get_job_details_handles_httperror_500(self, agent):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=500, text="Server Error"))
        agent.orchestrator_client.get.side_effect = http_error
        
        with pytest.raises(RuntimeError, match="HTTP error fetching job details.*500 - Server Error"):
            agent.get_job_details()

    def test_get_job_details_handles_requestexception(self, agent):
        agent.orchestrator_client.get.side_effect = requests.exceptions.RequestException("Network Error")
        
        with pytest.raises(RuntimeError, match="Request error fetching job details.*Network Error"):
            agent.get_job_details()

    def test_get_job_details_handles_invalid_json(self, agent):
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("JSON Decode Error")
        agent.orchestrator_client.get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Failed to decode JSON response.*JSON Decode Error"):
            agent.get_job_details()

    @patch.dict(os.environ, {"UIPATH_URL": ""}, clear=True) # No UIPATH_URL
    def test_get_job_details_raises_valueerror_if_uipath_url_missing(self, mock_uipath_retry_session):
        # Re-initialize agent within this specific env context
        agent_no_url = OrchestratorTroubleshootingAgent(jobKey="123") 
        agent_no_url.orchestrator_client = mock_uipath_retry_session
        agent_no_url.uipath_url = "" # Ensure it's empty on the instance too
        
        with pytest.raises(ValueError, match="UIPATH_URL is not configured"):
            agent_no_url.get_job_details()
    
    @pytest.mark.skipif(not UIPATH_SDK_AVAILABLE_FOR_TEST, reason="UiPath SDK not installed")
    def test_get_job_details_handles_uipatherror(self, agent):
        # This test will only run if UiPathError can be imported
        agent.orchestrator_client.get.side_effect = UiPathError("SDK Specific Error")
        
        with pytest.raises(RuntimeError, match="UiPath API error fetching job details.*SDK Specific Error"):
            agent.get_job_details()


class TestTroubleshootMethod:
    """Tests for the troubleshoot method's high-level flow."""

    @pytest.fixture
    def agent(self, mock_env_vars, mock_uipath_retry_session):
        # Agent with some default params for llm and slack
        return OrchestratorTroubleshootingAgent(
            jobKey="ts_job_1",
            llm_model_name="test_llm",
            slack_notification_channel="test_slack_channel"
        )

    @patch('uqlm.orchestrator_agent.invoke_uipath_llm')
    @patch('uqlm.orchestrator_agent.get_uipath_llm')
    @patch('uqlm.orchestrator_agent.BlackBoxUQ') # Assuming BlackBoxUQ is used
    def test_troubleshoot_flow_successful_details(
        self, mock_bb_uq, mock_get_llm, mock_invoke_llm, agent, capsys
    ):
        sample_job_details = {"Id": "ts_job_1", "State": "Faulted", "Info": "Test error"}
        agent.get_job_details = MagicMock(return_value=sample_job_details)
        agent.send_solution_to_slack = MagicMock()

        mock_llm_client = MagicMock()
        mock_get_llm.return_value = mock_llm_client
        mock_invoke_llm.return_value = "LLM solution: fix the error."
        
        mock_uq_scorer_instance = mock_bb_uq.return_value

        result = agent.troubleshoot()

        agent.get_job_details.assert_called_once()
        # TODO: Add asserts for LLM and UQ calls once those TODOs are implemented
        # mock_get_llm.assert_called_once_with(llm_name="test_llm")
        # mock_invoke_llm.assert_called_once_with(mock_llm_client, pytest.ANY)
        # mock_bb_uq.assert_called_once() # Or however it's initialized
        
        # Check that send_solution_to_slack was called
        agent.send_solution_to_slack.assert_called_once()
        # The first argument to send_solution_to_slack is the placeholder solution string
        assert "Troubleshooting for Job ID ts_job_1" in agent.send_solution_to_slack.call_args[0][0]
        assert agent.send_solution_to_slack.call_args[1]['slack_channel'] == "test_slack_channel"
        
        assert "Successfully fetched details" in result
        assert "Job State: Faulted" in result
        assert "Troubleshooting steps would be generated here using LLM and UQ." not in result # Since it's replaced by actual solution now
        assert "Generated Solution:" in result
        assert "fix the error" not in result # As LLM/UQ part is still placeholder in the string construction for now
        assert "(This is a placeholder solution generated by the agent.)" in result


    def test_troubleshoot_when_get_job_details_fails(self, agent):
        agent.get_job_details = MagicMock(side_effect=RuntimeError("Failed to fetch details"))
        agent.send_solution_to_slack = MagicMock()

        result = agent.troubleshoot()
        
        agent.get_job_details.assert_called_once()
        agent.send_solution_to_slack.assert_not_called() # Should not be called if details fail
        assert "An error occurred during troubleshooting" in result
        assert "Failed to fetch details" in result

    def test_troubleshoot_when_get_job_details_returns_error_string(self, agent):
        # Scenario where get_job_details catches its own error and returns a string
        agent.get_job_details = MagicMock(return_value="Custom error string from get_job_details")
        agent.send_solution_to_slack = MagicMock()

        result = agent.troubleshoot()
        
        agent.get_job_details.assert_called_once()
        agent.send_solution_to_slack.assert_not_called()
        assert result == "Custom error string from get_job_details"


class TestSendSolutionToSlack:
    """Tests for the send_solution_to_slack method."""

    @pytest.fixture
    def agent(self, mock_env_vars, mock_uipath_retry_session):
        return OrchestratorTroubleshootingAgent(jobKey="slack_test_job")

    def test_send_solution_to_slack_prints_placeholders_and_returns_true(self, agent, capsys):
        solution = "Step 1: Do this. Step 2: Do that."
        result = agent.send_solution_to_slack(solution_steps=solution, slack_channel="test-channel")
        
        captured = capsys.readouterr()
        assert "Attempting to send solution to Slack channel: test-channel" in captured.out
        assert f"Solution:\n{solution}" in captured.out
        assert "Placeholder: Slack integration not fully implemented. Simulating success." in captured.out
        assert result is True

    def test_send_solution_to_slack_channel_resolution_argument(self, agent, capsys):
        agent.send_solution_to_slack("solution", slack_channel="arg-channel")
        captured = capsys.readouterr()
        assert "Attempting to send solution to Slack channel: arg-channel" in captured.out

    @patch.dict(os.environ, {"SLACK_DEFAULT_CHANNEL": "env-channel"}, clear=True)
    def test_send_solution_to_slack_channel_resolution_env_var(self, agent_with_cleared_env_vars_fixture, capsys):
        # Need a new agent instance that will pick up the patched env var
        agent = OrchestratorTroubleshootingAgent(jobKey="slack_test_job_env")
        agent.send_solution_to_slack("solution") # No slack_channel argument
        captured = capsys.readouterr()
        assert "Attempting to send solution to Slack channel: env-channel" in captured.out

    @patch.dict(os.environ, {}, clear=True) # No SLACK_DEFAULT_CHANNEL
    def test_send_solution_to_slack_channel_resolution_default(self, agent_with_cleared_env_vars_fixture, capsys):
        agent = OrchestratorTroubleshootingAgent(jobKey="slack_test_job_default")
        agent.send_solution_to_slack("solution") # No arg, no env var
        captured = capsys.readouterr()
        assert "Attempting to send solution to Slack channel: default-general" in captured.out
