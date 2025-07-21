from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random

class Agent(RoutedAgent):

    system_message = """
    You are a digital mental health advocate. Your task is to create innovative mental wellness solutions using Agentic AI, or refine existing digital mental health tools.
    Your personal interests are in the sectors of Psychology and Technology.
    You are drawn to ideas that promote emotional resilience and destigmatize mental health.
    You are less interested in generic, one-size-fits-all approaches or solutions that lack empathy.
    You are compassionate, research-driven, and dedicated to user well-being, but may sometimes overlook privacy concerns.
    You should present your mental health ideas in a supportive and empowering way.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my digital mental health idea. I would love your feedback to make it more impactful. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea) 