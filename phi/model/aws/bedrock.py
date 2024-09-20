import json
from typing import Any, Dict, Iterator, List, Optional

from phi.aws.api_client import AwsApiClient
from phi.model.base import Model
from phi.model.message import Message
from phi.model.response import ModelResponse
from phi.utils.log import logger
from phi.utils.timer import Timer
from phi.utils.tools import get_function_call_for_tool_call

try:
    from boto3 import session  # noqa: F401
    from botocore.exceptions import ClientError
except ImportError:
    logger.error("`boto3` not installed")
    raise


class AwsBedrock(Model):
    name: str = "AwsBedrock"
    model: str

    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None
    aws_client: Optional[AwsApiClient] = None
    # -*- Request parameters
    request_params: Optional[Dict[str, Any]] = None

    _bedrock_client: Optional[Any] = None
    _bedrock_runtime_client: Optional[Any] = None

    def get_aws_region(self) -> Optional[str]:
        """
        Retrieves the AWS region to be used.

        Returns:
            Optional[str]: The AWS region if set, otherwise None.
        """
        # Priority 1: Use aws_region from model
        if self.aws_region is not None:
            return self.aws_region

        # Priority 2: Get aws_region from environment variable
        from os import getenv
        from phi.constants import AWS_REGION_ENV_VAR

        aws_region_env = getenv(AWS_REGION_ENV_VAR)
        if aws_region_env is not None:
            self.aws_region = aws_region_env
        return self.aws_region

    def get_aws_profile(self) -> Optional[str]:
        """
        Retrieves the AWS profile to be used.

        Returns:
            Optional[str]: The AWS profile if set, otherwise None.
        """
        # Priority 1: Use aws_profile from resource
        if self.aws_profile is not None:
            return self.aws_profile

        # Priority 2: Get aws_profile from environment variable
        from os import getenv
        from phi.constants import AWS_PROFILE_ENV_VAR

        aws_profile_env = getenv(AWS_PROFILE_ENV_VAR)
        if aws_profile_env is not None:
            self.aws_profile = aws_profile_env
        return self.aws_profile

    def get_aws_client(self) -> AwsApiClient:
        """
        Initializes and returns an AwsApiClient instance.

        Returns:
            AwsApiClient: An instance of AwsApiClient.
        """
        if self.aws_client is not None:
            return self.aws_client

        self.aws_client = AwsApiClient(
            aws_region=self.get_aws_region(), aws_profile=self.get_aws_profile()
        )
        return self.aws_client

    @property
    def bedrock_runtime_client(self) -> Any:
        """
        Initializes and returns the Bedrock runtime client.

        Returns:
            Any: The Bedrock runtime client.
        """
        if self._bedrock_runtime_client is not None:
            return self._bedrock_runtime_client

        boto3_session: session.Session = self.get_aws_client().boto3_session
        self._bedrock_runtime_client = boto3_session.client(
            service_name="bedrock-runtime"
        )
        return self._bedrock_runtime_client

    def invoke(
        self,
        modelId: str,
        messages: List[Dict[str, Any]],
        toolConfig: Optional[Dict[str, Any]] = None,
        system: Optional[List[Dict[str, Any]]] = None,
        inferenceConfig: Optional[Dict[str, Any]] = None,
        additionalModelRequestFields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invokes the Bedrock model with the given parameters.

        Args:
            modelId (str): The identifier of the model to invoke.
            messages (List[Dict[str, Any]]): The messages to send to the model.
            toolConfig (Optional[Dict[str, Any]], optional): Configuration for tools. Defaults to None.
            system (Optional[List[Dict[str, Any]]], optional): System prompts. Defaults to None.
            inferenceConfig (Optional[Dict[str, Any]], optional): Inference configuration. Defaults to None.
            additionalModelRequestFields (Optional[Dict[str, Any]], optional): Additional request fields. Defaults to None.

        Returns:
            Dict[str, Any]: The response from the model.
        """
        logger.debug(f"Making Bedrock request with modelId: {modelId}")
        request_params: Dict[str, Any] = {"modelId": modelId, "messages": messages}

        if toolConfig is not None:
            request_params["toolConfig"] = toolConfig

        if system is not None:
            request_params["system"] = system

        if inferenceConfig is not None:
            request_params["inferenceConfig"] = inferenceConfig

        if additionalModelRequestFields is not None:
            request_params["additionalModelRequestFields"] = (
                additionalModelRequestFields
            )

        return self.bedrock_runtime_client.converse(**request_params)

    def invoke_stream(
        self,
        modelId: str,
        messages: List[Dict[str, Any]],
        toolConfig: Optional[Dict[str, Any]] = None,
        system: Optional[List[Dict[str, Any]]] = None,
        inferenceConfig: Optional[Dict[str, Any]] = None,
        additionalModelRequestFields: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Invokes the Bedrock model in streaming mode.

        Args:
            modelId (str): The identifier of the model to invoke.
            messages (List[Dict[str, Any]]): The messages to send to the model.
            toolConfig (Optional[Dict[str, Any]], optional): Configuration for tools. Defaults to None.
            system (Optional[List[Dict[str, Any]]], optional): System prompts. Defaults to None.
            inferenceConfig (Optional[Dict[str, Any]], optional): Inference configuration. Defaults to None.
            additionalModelRequestFields (Optional[Dict[str, Any]], optional): Additional request fields. Defaults to None.

        Returns:
            Iterator[Dict[str, Any]]: An iterator over the streaming response.
        """
        logger.debug(f"Making Bedrock request with modelId: {modelId}")

        request_params: Dict[str, Any] = {"modelId": modelId, "messages": messages}

        if toolConfig is not None:
            request_params["toolConfig"] = toolConfig

        if system is not None:
            request_params["system"] = system

        if inferenceConfig is not None:
            request_params["inferenceConfig"] = inferenceConfig

        if additionalModelRequestFields is not None:
            request_params["additionalModelRequestFields"] = (
                additionalModelRequestFields
            )

        return self.bedrock_runtime_client.converse_stream(**request_params)

    def prepare_system_prompt(
        self, sys_prompt: Optional[str]
    ) -> Optional[List[Dict[str, str]]]:
        """
        Prepares the system prompt for the request.

        Args:
            sys_prompt (Optional[str]): The system prompt string.

        Returns:
            Optional[List[Dict[str, str]]]: A list containing the system prompt dictionary, or None.
        """
        if sys_prompt is not None:
            sys_prompt = [{"text": sys_prompt}]
            logger.debug(f"System prompt: {sys_prompt}")
        return sys_prompt

    def prepare_request_content(
        self, request_body: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extracts and prepares the request content from the request body.

        Args:
            request_body (Dict[str, Any]): The request body containing messages.

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries.
        """
        request_content: List[Dict[str, Any]] = request_body.get(
            "messages", [{"role": "user", "content": [{"text": None}]}]
        )
        return request_content

    def process_response(self, response: Dict[str, Any]) -> Message:
        """
        Processes the response from the model and creates an assistant message.

        Args:
            response (Dict[str, Any]): The response from the model invocation.

        Returns:
            Message: The assistant message generated from the response.
        """
        response_content: str = response["output"]["message"]["content"][0]["text"]
        response_role: str = response["output"]["message"]["role"]
        logger.debug(f"Response content: {response_content}")

        assistant_message = Message(
            role=response_role,
            content=response_content,
        )

        if response.get("stopReason") == "tool_use":
            tool_calls = self.extract_tool_calls(response)
            assistant_message.tool_calls = tool_calls
            logger.info(f"Assistant message with tool calls: {assistant_message}")

        return assistant_message

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts tool call information from the model's response.

        Args:
            response (Dict[str, Any]): The response from the model.

        Returns:
            List[Dict[str, Any]]: A list of tool call dictionaries.
        """
        tool_calls: List[Dict[str, Any]] = []
        for tool_use in response["output"]["message"]["content"]:
            if "toolUse" in tool_use:
                tool = tool_use["toolUse"]
                tool_name: str = tool["name"]
                tool_input: Dict[str, Any] = tool["input"]
                tool_use_id: str = tool["toolUseId"]

                logger.info(f"Tool request: {tool_name}. Input: {tool_input}")
                logger.info(f"Tool use ID: {tool_use_id}")

                function_def: Dict[str, Any] = {"name": tool_name}
                if tool_input:
                    function_def["arguments"] = json.dumps(tool_input)
                tool_calls.append(
                    {
                        "type": "function",
                        "function": function_def,
                        "tool_use_id": tool_use_id,
                    }
                )
        return tool_calls

    def update_usage_metrics(
        self, assistant_message: Message, response_timer: Timer
    ) -> None:
        """
        Updates the usage metrics based on the assistant's response and timing.

        Args:
            assistant_message (Message): The assistant message containing metrics.
            response_timer (Timer): The timer tracking response time.
        """
        # Add response time to metrics
        assistant_message.metrics["time"] = response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(response_timer.elapsed)

        # Token usage (placeholder values; replace with actual values if available)
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = prompt_tokens + completion_tokens

        assistant_message.metrics["prompt_tokens"] = prompt_tokens
        assistant_message.metrics["completion_tokens"] = completion_tokens
        assistant_message.metrics["total_tokens"] = total_tokens

        self.metrics["prompt_tokens"] = (
            self.metrics.get("prompt_tokens", 0) + prompt_tokens
        )
        self.metrics["completion_tokens"] = (
            self.metrics.get("completion_tokens", 0) + completion_tokens
        )
        self.metrics["total_tokens"] = (
            self.metrics.get("total_tokens", 0) + total_tokens
        )

    def handle_tool_calls(
        self,
        messages: List[Message],
        assistant_message: Message,
        model_response: ModelResponse,
    ) -> ModelResponse:
        """
        Handles tool calls by executing the functions and updating the response.

        Args:
            messages (List[Message]): The list of conversation messages.
            assistant_message (Message): The assistant's message containing tool calls.
            model_response (ModelResponse): The model's response to be updated.

        Returns:
            ModelResponse: The updated model response after handling tool calls.
        """
        # Prepare for function calls
        model_response.content = f"{assistant_message.content}\n\n"
        function_calls_to_run: List[Any] = []
        tool_ids: List[str] = [
            tool_call["tool_use_id"] for tool_call in assistant_message.tool_calls
        ]

        for tool_call in assistant_message.tool_calls:
            function_call = get_function_call_for_tool_call(tool_call, self.functions)
            if function_call is None:
                messages.append(
                    Message(role="user", content="Could not find function to call.")
                )
                continue
            if function_call.error is not None:
                messages.append(Message(role="user", content=function_call.error))
                continue
            function_calls_to_run.append(function_call)

        # Show tool calls if enabled
        if self.show_tool_calls:
            model_response.content += self.format_tool_calls(function_calls_to_run)

        # Run function calls
        function_call_results: List[Message] = self.run_function_calls(
            function_calls_to_run
        )
        if function_call_results:
            fc_responses: List[Dict[str, Any]] = self.prepare_function_call_responses(
                function_call_results, tool_ids
            )
            try:
                messages.append(Message(role="user", content=json.dumps(fc_responses)))
            except json.JSONDecodeError as e:
                logger.error(f"Error serializing fc_responses: {e}")
                messages.append(
                    Message(role="user", content=str(fc_responses))
                )  # Fallback to string representation

        # Get new response after tool calls
        response_after_tool_calls = self.response(messages=messages)
        if response_after_tool_calls.content:
            model_response.content += response_after_tool_calls.content

        return model_response

    def format_tool_calls(self, function_calls_to_run: List[Any]) -> str:
        """
        Formats the function calls for display or logging.

        Args:
            function_calls_to_run (List[Any]): The list of function calls to be run.

        Returns:
            str: A formatted string representing the function calls.
        """
        if len(function_calls_to_run) == 1:
            return f" - Running: {function_calls_to_run[0].get_call_str()}\n\n"
        elif len(function_calls_to_run) > 1:
            calls_str = "Running:"
            for func_call in function_calls_to_run:
                calls_str += f"\n - {func_call.get_call_str()}"
            return f"{calls_str}\n\n"
        return ""

    def prepare_function_call_responses(
        self, function_call_results: List[Message], tool_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Prepares the responses from function calls to be sent back to the assistant.

        Args:
            function_call_results (List[Message]): The results from the function calls.
            tool_ids (List[str]): The IDs of the tools used.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing tool results.
        """
        fc_responses: List[Dict[str, Any]] = []
        for idx, fc_message in enumerate(function_call_results):
            fc_responses.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_ids[idx],
                    "content": fc_message.content,
                }
            )
        return fc_responses

    def response(self, messages: List[Message]) -> ModelResponse:
        """
        Generates a response based on the given messages.

        Args:
            messages (List[Message]): The conversation messages.

        Returns:
            ModelResponse: The model's response.
        """

        # Log the start of the response generation process for debugging purposes
        logger.debug("---------- Bedrock Response Start ----------")

        # Iterate over each message in the conversation history and log it for debugging
        for m in messages:
            m.log()

        # Initialize a new ModelResponse object to store the model's response
        model_response = ModelResponse()

        # Create and start a timer to measure how long it takes to generate the response
        response_timer = Timer()
        response_timer.start()

        # Prepare the request body by converting the messages into the format expected by the model
        request_body: Dict[str, Any] = self.get_request_body(messages)
        logger.debug(f"Request body: {request_body}")

        # Extract any tools specified in the request body, if available
        tools = request_body.get("tools")
        logger.debug(f"Tools: {tools}")

        # Prepare the system prompt if one is provided in the request body
        sys_prompt = self.prepare_system_prompt(request_body.get("system"))

        # Prepare the content of the request, including user messages and any additional context
        request_content = self.prepare_request_content(request_body)
        logger.debug(f"Request Content: {request_content}")

        # Invoke the model with the prepared request content, system prompt, and tool configuration
        response: Dict[str, Any] = self.invoke(
            modelId=self.model,
            messages=request_content,
            toolConfig={"tools": tools} if tools else None,
            system=sys_prompt if sys_prompt else None,
        )

        # Stop the response timer and log how long it took to generate the response
        response_timer.stop()
        logger.debug(f"Time to generate response: {response_timer.elapsed:.4f}s")
        logger.debug(f"Response: {response}")

        # Process the model's response to extract the assistant's message and any relevant information
        assistant_message: Message = self.process_response(response)

        # Add the assistant's message to the conversation history
        messages.append(assistant_message)

        # Log the assistant's message for debugging purposes
        assistant_message.log()

        # Update usage metrics such as token counts and timing information
        self.update_usage_metrics(assistant_message, response_timer)

        # If the assistant's message includes any tool calls and running tools is enabled,
        # handle the tool calls and return the updated model response
        if assistant_message.tool_calls and self.run_tools:
            return self.handle_tool_calls(messages, assistant_message, model_response)

        # If the assistant's message contains content, set it as the content of the model response
        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()

        # Log the end of the response generation process
        logger.debug("---------- Bedrock Response End ----------")
        logger.debug(messages)

        # Return the final model response to the caller
        return model_response

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        """
        Generates a streaming response based on the given messages.

        Args:
            messages (List[Message]): The conversation messages.

        Yields:
            Iterator[ModelResponse]: An iterator over model responses.
        """

        # Log the start of the streaming response generation process for debugging purposes
        logger.debug("---------- Bedrock Response Start ----------")

        # Create and start a timer to measure how long it takes to generate the response
        response_timer = Timer()
        response_timer.start()

        # Prepare the request body by converting the messages into the format expected by the model
        request_body: Dict[str, Any] = self.get_request_body(messages)
        logger.debug(f"Request body: {request_body}")

        # Extract any tools specified in the request body, if available
        tools = request_body.get("tools", None)
        logger.debug(f"Tools: {tools}")

        # Extract and prepare the system prompt if provided
        sys_prompt = request_body.get("system", None)
        if sys_prompt is not None:
            # Format the system prompt as expected by the model
            sys_prompt = [{"text": sys_prompt}]
            logger.debug(f"System prompt: {sys_prompt}")

        # Extract the content of the messages to be sent to the model
        request_content = request_body.get("messages", None)
        logger.debug(f"Request Content: {request_content}")

        # Invoke the model's streaming interface with the prepared request content, system prompt, and tool configuration
        response = self.invoke_stream(
            modelId=self.model,
            messages=request_content,
            toolConfig={"tools": tools} if tools else None,
            system=sys_prompt,
        )

        # Initialize variables to accumulate the assistant's message content and token usage metrics
        assistant_message_content: str = ""
        response_prompt_tokens: int = 0
        response_completion_tokens: int = 0

        # Retrieve the streaming response from the model
        stream = response.get("stream")
        if stream:
            # Iterate over each event in the streaming response
            for event in stream:
                # If the event contains a chunk of the assistant's message
                if "contentBlockDelta" in event:
                    message_content: str = event["contentBlockDelta"]["delta"]["text"]
                    if message_content is not None:
                        # Yield the current chunk of the assistant's message
                        yield ModelResponse(content=message_content)
                        # Accumulate the assistant's message content
                        assistant_message_content += message_content
                # If the event contains metadata, such as token usage information
                if "metadata" in event:
                    metadata = event["metadata"]
                    if "usage" in metadata:
                        # Accumulate the token usage counts from the metadata
                        response_prompt_tokens += metadata["usage"]["inputTokens"]
                        response_completion_tokens += metadata["usage"]["outputTokens"]

            # Stop the response timer and log how long it took to generate the response
            response_timer.stop()
            logger.debug(f"Time to generate response: {response_timer.elapsed:.4f}s")

        # Calculate the total number of tokens used in the response
        total_tokens: int = response_prompt_tokens + response_completion_tokens

        # Create a new assistant message with the accumulated content and usage metrics
        assistant_message = Message(
            role="assistant",
            content=(
                assistant_message_content if assistant_message_content != "" else None
            ),
            metrics={
                "time_to_first_token": None,  # Placeholder for future implementation
                "time_per_output_token": (
                    f"{response_timer.elapsed / response_completion_tokens:.4f}s"
                    if response_completion_tokens > 0
                    else None
                ),
                "prompt_tokens": response_prompt_tokens,
                "completion_tokens": response_completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        # Add the assistant's message to the conversation history
        messages.append(assistant_message)

        # Log the assistant's message for debugging purposes
        assistant_message.log()

        # Log the end of the streaming response generation process
        logger.debug("---------- Bedrock Response End ----------")
        logger.debug(messages)
