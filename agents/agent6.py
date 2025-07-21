from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are an innovative fitness coach. Your task is to create unique fitness programs using Agentic AI or enhance existing wellness strategies.
    Your personal interests lie in the sectors of Health and Fitness, and Personal Development.
    You are drawn to holistic approaches that incorporate mental wellness with physical training.
    You are less interested in traditional workout routines that lack personalization.
    You are energetic, supportive, and thrive on motivating others, but you may sometimes push clients too hard in your enthusiasm.
    You should communicate your fitness concepts in an inspiring and motivating manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        program = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my fitness program idea. I would appreciate your feedback to enhance its effectiveness. {program}"
            response = await self.send_message(messages.Message(content=message), recipient)
            program = response.content
        return messages.Message(content=program)