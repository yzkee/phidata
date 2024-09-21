import logging
import json
import boto3

from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StationNotFoundError(Exception):
    """Raised when a radio station isn't found."""

    pass


def get_top_song(call_sign):
    """Returns the most popular song for the requested station.
    Args:
        call_sign (str): The call sign for the station for which you want
        the most popular song.

    Returns:
        response (json): The most popular song and artist.
    """

    song = ""
    artist = ""
    if call_sign == "WZPZ":
        song = "Elemental Hotel"
        artist = "8 Storey Hike"

    else:
        raise StationNotFoundError(f"Station {call_sign} not found.")

    return song, artist


def stream_messages(bedrock_client, model_id, messages, tool_config):
    """
    Sends a message to a model and streams the response.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        messages (JSON) : The messages to send to the model.
        tool_config : Tool Information to send to the model.

    Returns:
        stop_reason (str): The reason why the model stopped generating text.
        message (JSON): The message that the model generated.

    """

    logger.info("Streaming messages with model %s", model_id)

    response = bedrock_client.converse_stream(
        modelId=model_id, messages=messages, toolConfig=tool_config
    )

    stop_reason = ""

    message = {}
    content = []
    message["content"] = content
    text = ""
    tool_use = {}

    # stream the response into a message.
    for chunk in response["stream"]:
        if "messageStart" in chunk:
            message["role"] = chunk["messageStart"]["role"]
        elif "contentBlockStart" in chunk:
            tool = chunk["contentBlockStart"]["start"]["toolUse"]
            tool_use["toolUseId"] = tool["toolUseId"]
            tool_use["name"] = tool["name"]
        elif "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "toolUse" in delta:
                if "input" not in tool_use:
                    tool_use["input"] = ""
                tool_use["input"] += delta["toolUse"]["input"]
            elif "text" in delta:
                text += delta["text"]
                print(delta["text"], end="")
        elif "contentBlockStop" in chunk:
            if "input" in tool_use:
                tool_use["input"] = json.loads(tool_use["input"])
                content.append({"toolUse": tool_use})
                tool_use = {}
            else:
                content.append({"text": text})
                text = ""

        elif "messageStop" in chunk:
            stop_reason = chunk["messageStop"]["stopReason"]

    return stop_reason, message


def main():
    """
    Entrypoint for streaming tool use example.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    input_text = "What is the most popular song on WZPZ?"

    try:
        bedrock_client = boto3.client(service_name="bedrock-runtime")

        # Create the initial message from the user input.
        messages = [{"role": "user", "content": [{"text": input_text}]}]

        # Define the tool to send to the model.
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ]
        }

        # Send the message and get the tool use request from response.
        stop_reason, message = stream_messages(
            bedrock_client, model_id, messages, tool_config
        )
        messages.append(message)

        if stop_reason == "tool_use":

            for content in message["content"]:
                if "toolUse" in content:
                    tool = content["toolUse"]

                    if tool["name"] == "top_song":
                        tool_result = {}
                        try:
                            song, artist = get_top_song(tool["input"]["sign"])
                            tool_result = {
                                "toolUseId": tool["toolUseId"],
                                "content": [{"json": {"song": song, "artist": artist}}],
                            }
                        except StationNotFoundError as err:
                            tool_result = {
                                "toolUseId": tool["toolUseId"],
                                "content": [{"text": err.args[0]}],
                                "status": "error",
                            }

                        tool_result_message = {
                            "role": "user",
                            "content": [{"toolResult": tool_result}],
                        }
                        # Add the result info to message.
                        messages.append(tool_result_message)

        # Send the messages, including the tool result, to the model.
        stop_reason, message = stream_messages(
            bedrock_client, model_id, messages, tool_config
        )

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))

    else:
        print(f"\nFinished streaming messages with model {model_id}.")


if __name__ == "__main__":
    main()
