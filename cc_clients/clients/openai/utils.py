from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, CompletionUsage

DUMMY_CHAT_COMPLETION = ChatCompletion(
    id="chatcmpl-123",
    choices=[Choice(
        index=0,
        finish_reason="stop",
        message=ChatCompletionMessage(
            content="dummy_content",
            role="assistant",
        ),
    )],
    created=0,
    model="dummy_model",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    ),
)


def make_valid_response(response: ChatCompletion, **kwargs) -> ChatCompletion:
    response.object = "chat.completion"
    response.id = "dummy_id" if response.id is None else response.id
    response.created = 0 if response.created is None else response.created
    response.model = kwargs.get("model") if response.model is None else response.model
    return response
