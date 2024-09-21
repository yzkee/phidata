import json
from typing import Any, Dict, Iterator, List, Optional, Union

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

        # Token usage (use the values directly from assistant_message.metrics)
        prompt_tokens: int = assistant_message.metrics.get("prompt_tokens", 0)
        completion_tokens: int = assistant_message.metrics.get("completion_tokens", 0)
        total_tokens: int = assistant_message.metrics.get("total_tokens", 0)

        # Update overall metrics
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
        stream: bool = False,
        sys_prompt: Optional[str] = None,
    ) -> Iterator[ModelResponse]:
        """
        Handles tool calls by executing the functions and updating the response.

        Args:
            messages (List[Message]): The list of conversation messages.
            assistant_message (Message): The assistant's message containing tool calls.
            model_response (ModelResponse): The model's response to be updated.
            stream (bool): Whether to stream the tool calls.
            sys_prompt (Optional[str]): The system prompt to be used.

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
        if not stream:
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
        # This determines the response content, tool calls, and metrics
        assistant_message: Message = self.process_response(response)

        # Extract usage data from the response and unpack into assistant_message.metrics
        usage = response.get("usage", {})
        assistant_message.metrics.update(
            {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
        )

        # Add the assistant's message to the conversation history
        messages.append(assistant_message)

        # Log the assistant's message for debugging purposes
        assistant_message.log()

        # Update usage metrics such as token counts and timing information
        self.update_usage_metrics(assistant_message, response_timer)

        # If the assistant's message includes any tool calls and running tools is enabled,
        # handle the tool calls and return the updated model response
        if assistant_message.tool_calls and self.run_tools:
            logger.debug(f"Assistant message has tool calls: {assistant_message}")
            return self.handle_tool_calls(messages, assistant_message, model_response, stream=False, sys_prompt=sys_prompt)

        # If the assistant's message contains content, set it as the content of the model response
        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()

        # Log the end of the response generation process
        logger.debug("---------- Bedrock Response End ----------")
        logger.debug(messages)

        # Return the final model response to the caller
        return model_response

    def create_final_message(self, messages: List[Message]) -> List[Dict[str, Any]]:
        # Send the messages to the model to get the final response
        final_messages = []
        # skipping the first message because it is the system prompt
        for i in messages[1:]:
            roles = i.role
            content = i.content
            if isinstance(content, list) and all(isinstance(item, dict) and 'text' in item for item in content):
                # Content is already in the desired format
                pass
            elif isinstance(content, str):
                try:
                    # Try to parse the content as JSON
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, list):
                        # Handle list of parsed items
                        new_content = []
                        for item in parsed_content:
                            if isinstance(item, dict) and 'content' in item:
                                # If 'content' is a JSON string, parse it
                                inner_content = item['content']
                                try:
                                    inner_parsed_content = json.loads(inner_content)
                                    if isinstance(inner_parsed_content, list):
                                        for inner_item in inner_parsed_content:
                                            new_content.append({'text': json.dumps(inner_item)})
                                    else:
                                        new_content.append({'text': str(inner_parsed_content)})
                                except json.JSONDecodeError:
                                    new_content.append({'text': inner_content})
                            else:
                                new_content.append({'text': str(item)})
                        content = new_content
                    else:
                        content = [{"text": str(parsed_content)}]
                except json.JSONDecodeError:
                    content = [{"text": content}]
            else:
                content = [{"text": str(content)}]
            final_messages.append({"role": roles, "content": content})        
        return final_messages
    
    def create_message_from_stream_result(self, response: Dict[str, Any]) -> Message:
        """
        Processes the response from the model and creates an assistant message.
        Args:
            response (Dict[str, Any]): The response from the model invocation.
        Returns:
            Message: The assistant message generated from the response.
        """
        content = response.get("content", [])
        role = response.get("role", "")
        
        response_content: Union[List[Dict], str] = []
        tool_calls: List[Dict[str, Any]] = []

        for item in content:
            if "text" in item:
                if isinstance(response_content, list):
                    response_content.append({"text": item["text"]})
            elif "toolUse" in item:
                tool_use = item["toolUse"]
                tool_call = {
                    "tool_use_id": tool_use.get("toolUseId", ""),
                    "type": "function",
                    "function": {
                        "name": tool_use.get("name", ""),
                        "arguments": json.dumps(tool_use.get("input", {}))
                    }
                }
                tool_calls.append(tool_call)

        logger.debug(f"Response content: {response_content}")
        
        assistant_message = Message(
            role=role,
            content=response_content,
            tool_calls=tool_calls if tool_calls else None
        )
        return assistant_message

    def stream_messages(self,
                        model_id: str,
                        messages: List[Dict[str, Any]],
                        tool_config: Optional[Dict[str, Any]] = None,
                        sys_prompt: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Sends a message to a model and streams the response.
        Args:
            bedrock_client: The Boto3 Bedrock runtime client.
            model_id (str): The model ID to use.
            messages (JSON) : The messages to send to the model.
            tool_config : Tool Information to send to the model.
            sys_prompt : System prompt to send to the model.

        Returns:
            stop_reason (str): The reason why the model stopped generating text.
            message (JSON): The message that the model generated.
            usage (JSON): The usage metrics for the model.

        """

        logger.info("Streaming messages with model %s", model_id)

        request_params = {"modelId": model_id, "messages": messages}

        if tool_config is not None:
            request_params["toolConfig"] = {"tools": tool_config}
        
        if sys_prompt is not None:
            request_params["system"] = sys_prompt

        response = self.bedrock_runtime_client.converse_stream(**request_params)
        

        stop_reason = ""
    
        message = {}
        content = []
        message['content'] = content
        text = ''
        tool_use = {}


        # stream the response into a message.
        for chunk in response['stream']:
            if 'messageStart' in chunk:
                message['role'] = chunk['messageStart']['role']
            elif 'contentBlockStart' in chunk:
                tool = chunk['contentBlockStart']['start']['toolUse']
                tool_use['toolUseId'] = tool['toolUseId']
                tool_use['name'] = tool['name']
            elif 'contentBlockDelta' in chunk:
                delta = chunk['contentBlockDelta']['delta']
                if 'toolUse' in delta:
                    if 'input' not in tool_use:
                        tool_use['input'] = ''
                    tool_use['input'] += delta['toolUse']['input']
                elif 'text' in delta:
                    text += delta['text']
                    yield ModelResponse(content=delta['text'])
            elif 'contentBlockStop' in chunk:
                if 'input' in tool_use:
                    tool_use['input'] = json.loads(tool_use['input'])
                    content.append({'toolUse': tool_use})
                    tool_use = {}
                else:
                    content.append({'text': text})
                    text = ''
            elif 'metadata' in chunk:
                usage = chunk['metadata']['usage']

            elif 'messageStop' in chunk:
                stop_reason = chunk['messageStop']['stopReason']

        # return the results of the stream as a dictionary
        yield {'stop_reason': stop_reason, 'message': message, 'usage': usage}

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        """
        Generates a streaming response based on the given messages.

        If a tool call is detected, it will preform single turn tool use and return the results

        Args:
            messages (List[Message]): The conversation messages.

        Yields:
            Iterator[ModelResponse]: An iterator over model responses.
        """
        model_response = ModelResponse()
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
        sys_prompt = self.prepare_system_prompt(request_body.get("system"))
        logger.debug(f"System prompt: {sys_prompt}")

        # Extract the content of the messages to be sent to the model
        request_content = self.prepare_request_content(request_body)

        logger.debug(f"Request Content: {request_content}")

        # Stream the initial messages from the model
        for response_chunk in self.stream_messages(self.model, request_content, tools, sys_prompt):
                if isinstance(response_chunk, ModelResponse):
                    # Yield the ModelResponse objects as they come
                    yield response_chunk
                else:
                    # Handle the final stop_reason, message, and usage
                    stop_reason = response_chunk.get('stop_reason')
                    message = response_chunk.get('message')
                    usage = response_chunk.get('usage')

        # Create assistant message from the streamed response
        assistant_message = self.create_message_from_stream_result(message)
        assistant_message.metrics.update(
            {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
        )
        assistant_message.log()
        messages.append(assistant_message)

        # Check if the stop reason is tool use, if so handle tool calls (run tools and add results to message)
        if assistant_message.tool_calls and self.run_tools and stop_reason == "tool_use":
           # This will update Messages with the tool call results
           self.handle_tool_calls(messages, assistant_message, model_response, stream=True, sys_prompt=sys_prompt)

        # create the final messages to send to the model
        final_messages = self.create_final_message(messages)

        # Stream the final messages from the model
        logger.debug(f"Final messages: {final_messages}")
        for response_chunk in self.stream_messages(self.model, final_messages, sys_prompt=sys_prompt):
                if isinstance(response_chunk, ModelResponse):
                    # Yield the ModelResponse objects as they come
                    yield response_chunk
                else:
                    # Handle the final stop_reason and message
                    stop_reason = response_chunk.get('stop_reason')
                    message = response_chunk.get('message')
                    usage = response_chunk.get('usage')

        # Create assistant message from the final streamed response
        assistant_message = self.create_message_from_stream_result(message)
        assistant_message.metrics.update(
            {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
        )
        assistant_message.log()
        messages.append(assistant_message)

        logger.debug(f"Final messages: {messages}")

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()       

        yield from self.response_stream(messages=messages)